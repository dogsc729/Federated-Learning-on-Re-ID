from configparser import Interpolation
from tracemalloc import start
import torch
import torch.nn as nn
import os
import copy
from torchvision import datasets, transforms
import torchvision.models as models
from torch.hub import load_state_dict_from_url
import numpy as np
import time
import sys
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
import re
import argparse
from torchsummary import summary

from update import LocalUpdate
from models import Global_model, Client_model
from utils import average_weights, selective_aggregation
from test_big_dataset import fliplr, load_network, extract_feature, get_id, evaluate, compute_mAP

'''
argument parsing
'''
def get_parser():
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument('-s', '--scenario', default = 'ska', type = str)
    parser.add_argument('-l', '--location', default = '', type = str)
    parser.add_argument('--global_iter', default = '100', type = int)
    parser.add_argument('--local_epoch', default = '1', type = int)
    parser.add_argument('--lr_feature', default = '0.01', type = float)
    parser.add_argument('--lr_classifier', default = '0.1', type = float)
    parser.add_argument('-m', '--model', default = 'attentive', type = str)
    return parser

'''
For Recording log
'''
class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
'''
Start from here
'''            
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('Scenario: ' + args.scenario)
    print('Location: ' + args.location)
    print('Model: ' + args.model)

    scenario = args.scenario
    if (args.scenario != 'ska' and args.scenario != 'fed'):
        raise ValueError("Scenario must be one of ['ska', 'fed']")

    model_type = args.model
    if args.model == 'attentive':
        import models_ska as resnets
    elif args.model == 'vanilla':    
        import resnet_vallina as resnets
    elif args.model == 'ibn':
        import resnet_ibn_a as resnets
    else:
        raise ValueError("Model type must be one of ['attentive', 'vanilla', 'ibn']")

    '''
    Checkpoint Location 
    '''
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    if args.location == '':
        checkpoint_dir = "../checkpoint/" + current_time
        log_path = checkpoint_dir + "/" + current_time + ".log"
    else:
        checkpoint_dir = "../checkpoint/" + args.location
        log_path = checkpoint_dir + "/" + args.location + ".log"
    if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

    sys.stdout = Logger(log_path)

    '''
    Setup for evaluation phase
    '''
    data_transforms = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_dir = "../datasets/Market/pytorch"

    test_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    test_dataloaders = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=128,
                                                shuffle=False, num_workers=8) for x in ['gallery','query']}

    class_names = test_datasets['query'].classes

    gallery_path = test_datasets['gallery'].imgs
    query_path = test_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    '''
    Configs
    '''
    config = {
        "global_iteration": args.global_iter,
        "local_optimization_epoch": args.local_epoch,
        "lr_feature_extraction": args.lr_feature,
        "lr_classifier": args.lr_classifier,
        "batch_size": 32,
        "warm_epoch": 10,
    }

    '''
    Setting Seed
    '''
    myseed = 7414  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    '''
    Transforms
    '''
    train_transform_list = [
        transforms.Resize((256,128)),
        transforms.Pad(10), # optional
        transforms.RandomCrop((256, 128)), # optional
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    valid_transform_list = [
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_transform = transforms.Compose(train_transform_list)
    valid_transform = transforms.Compose(valid_transform_list)

    '''
    Setup for Training and validation phase 
    '''
    dataset_list = ['MSMT17','cuhk03-np-detected','DukeMTMC-reID', 'Market', ]
    #dataset_list = ['cuhk03-np-detected','Market', ]
    #dataset_list = ['Market']
    dataloader_dict = {}
    #ID_size_dict = {'MSMT17': 2854, 'cuhk03-np-detected': 2580, 'DukeMTMC-reID': 2515, 'Market': 2564} # after adding CUHK02
    ID_size_dict = {'MSMT17': 1041, 'cuhk03-np-detected': 767, 'DukeMTMC-reID': 702, 'Market': 751}
    #train_img_size = [35206, 10224, 19446, 15811] # after adding CUHK02
    train_img_size = [31580, 6598, 15820, 12185]
    #train_img_size = [6598, 12185]
    #CUHK02 train ID size = 1813

    for dataset in dataset_list:
        data_dir = f'../datasets/{dataset}/pytorch'

        image_datasets = {}
        image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=train_transform) # RGB or BGR?

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = config['batch_size'], 
                    shuffle = True, num_workers = 8, pin_memory = True) for x in ['train', 'val']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        print("***************")
        print(dataset)
        print("train size:", dataset_sizes['train'])
        print("val size:", dataset_sizes['val'])
        #print(len(image_datasets['train'].classes)) # how many classes are there in training set.
        dataloader_dict[dataset] = dataloaders
    print("***************")
    print("==============> Dataloader Set Up Success")

    '''
    CUDA
    '''
    if torch.cuda.is_available():
        print("==============> Using GPU")
        device = 'cuda:0'
    else:
        print("==============> Using CPU")
        device = 'cpu'
        
    '''
    Load pretrained model of ResNet50 first
    '''
    model_urls = {
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    }

    state_dict = load_state_dict_from_url(model_urls['resnet50'])
    state_dict_keys = [k for k, v in state_dict.items()] # keys of vanilla resnet50

    '''
    Global model
    '''
    if args.model == "attentive" or args.model == "vanilla":
        global_model = resnets.__dict__['resnet50']()
        global_model.load_state_dict(state_dict, strict=False)
        if model_type == 'attentive':
            checkpoint = torch.load("model_best.pth.tar", map_location = 'cpu')
            st_dict = {k[15:]: v for k, v in checkpoint['state_dict'].items()} # dict of customize resnet50
            st_dict_keys = [k[15:] for k, v in checkpoint['state_dict'].items()] # keys of customize resnet50

            for key in state_dict_keys:
                try:
                    del st_dict[key] # remove the keys that are both in customize and vanilla resnet50
                except:
                    pass
            for k, v in st_dict.items():
                if re.findall("num_batches_tracked", k):
                    st_dict[k] = torch.tensor(0)

            global_model.load_state_dict(st_dict, strict=False)
    elif args.model == "ibn":
        global_model = resnets.__dict__['resnet50_ibn_a'](last_stride=1, pretrained=True)

    global_model.to(device)
    global_model.train()

    # Copy Weights
    global_weights = global_model.state_dict()


    criterion = nn.CrossEntropyLoss()
    n_epochs = config['global_iteration']
    local_epoch = config['local_optimization_epoch']
    local_model = {}

    '''
    Local models
    '''
    for dataset in dataset_list:
        local_model[dataset] = LocalUpdate(
                        device = device, 
                        criterion = criterion, 
                        dataloaders = dataloader_dict[dataset], 
                        output_dim = ID_size_dict[dataset],
                        local_epoch = local_epoch,
                        total_global_epoch = config["global_iteration"],
                        scenario = scenario,
                        model_type = model_type
                        )

    print("==============> Local Client Set Up Success")

    '''
    Scheduler
    '''
    num_warmup_epoch = config["warm_epoch"]
    total_global_epoch = config["global_iteration"]
    #warm_up_learning_rate = lambda current_epoch: 0.1 * float(current_epoch+1) / float(max(1, num_warmup_epoch)) if current_epoch < num_warmup_epoch \
                                        #else 1 if current_epoch < 40 else 0.1 if current_epoch < 80 else 0.01
    warm_up_learning_rate = lambda current_epoch: float(current_epoch+1)/ float(max(1, num_warmup_epoch)) if current_epoch < num_warmup_epoch \
        else max(0.0, float(total_global_epoch - current_epoch) / float(max(1, total_global_epoch - num_warmup_epoch)))

    scheduler_list = []                                   
    for dataset in dataset_list:
        #specific_scheduler = torch.optim.lr_scheduler.LambdaLR(local_model[dataset].specific_optimizer, lr_lambda = warm_up_learning_rate)
        #generalized_scheduler = torch.optim.lr_scheduler.LambdaLR(local_model[dataset].generalized_optimizer, lr_lambda = warm_up_learning_rate)
        #specific_scheduler = torch.optim.lr_scheduler.StepLR(local_model[dataset].specific_optimizer, step_size=40, gamma=0.1)
        #generalized_scheduler = torch.optim.lr_scheduler.StepLR(local_model[dataset].generalized_optimizer, step_size=40, gamma=0.1)
        specific_scheduler = torch.optim.lr_scheduler.MultiStepLR(local_model[dataset].specific_optimizer, milestones=[40,70], gamma=0.1)
        generalized_scheduler = torch.optim.lr_scheduler.MultiStepLR(local_model[dataset].generalized_optimizer, milestones=[40,70], gamma=0.1)
        scheduler_list.append(specific_scheduler)
        scheduler_list.append(generalized_scheduler)

    '''
    Setup Plotting the training process
    '''
    global_train_acc_list = []
    global_train_loss_list = []
    global_valid_acc_list = []
    global_valid_loss_list = []
    eval_r1_list = []
    eval_map_list = []
    epoch_list = []
    eval_epoch_list = []
    fig, axs = plt.subplots(2)
    axs[0].set_xlim([1, 100])
    axs[0].set_ylim([0.0, 1.0])
    axs[0].title.set_text("Train and Valid Acc")
    axs[1].set_xlim([1, 100])
    axs[1].set_ylim([0.0, 1.0])
    axs[1].title.set_text("Eval Acc")

    '''
    Start training
    '''
    best_map = 0.0
    best_r1 = 0.0
    for epoch in range(n_epochs):
        for scheduler in scheduler_list:
            print("Current Learning Rate: ", scheduler.get_last_lr())
        #checkpoint_path = checkpoint_dir + "/" + str(epoch+1) + ".ckpt"
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        for dataset in dataset_list:
            print("==========================================")
            print(f"Currently training on Client of {dataset}")
            print("==========================================")
            if scenario == 'ska':
                specific_weight, generalized_weight, loss, valid_loss, train_acc, valid_acc = local_model[dataset].update_weights()
                specific_weight = {k:v for k, v in specific_weight.items() if k in global_model.state_dict()} # choose only feature attraction layer
                generalized_weight = {k:v for k, v in generalized_weight.items() if k in global_model.state_dict()} # choose only feature attraction layer
            else:
                generalized_weight, loss, valid_loss, train_acc, valid_acc = local_model[dataset].update_weights()
                generalized_weight = {k:v for k, v in generalized_weight.items() if k in global_model.state_dict()} # choose only feature attraction layer
            '''
            Selective Aggregation
            '''
            if scenario == 'ska':
                selective_aggregation(specific_weight, generalized_weight, local_model[dataset].local_aggregated_model)

                aggregated_weight = local_model[dataset].local_aggregated_model.state_dict()

                local_weights.append(copy.deepcopy(aggregated_weight)) # the list of local weights of all clients
                #local_losses.append(copy.deepcopy(loss))
            else:
                local_weights.append(copy.deepcopy(generalized_weight)) # the list of local weights of all clients

        global_weights = average_weights(local_weights, train_img_size)

        global_model.load_state_dict(global_weights, strict=True)
        #torch.save(global_model.state_dict(), checkpoint_path)
        
        bckup_global_state_dict = global_model.state_dict() # save the global model for later reloading
        
        if ((epoch+1) % 5 == 0):
            '''
            Evaluation (every 5 Global epochs)
            '''
            if scenario == 'ska':
                specific_state_dict = local_model["Market"].specific_model.state_dict() # get the state dict of specific model of Market client
            else:
                specific_state_dict = local_model["Market"].generalized_model.state_dict() # Fedavg
            global_model.load_state_dict(specific_state_dict, strict = False) # load the state dict of specific model of Market for evaluation
            global_model.to(device)
            global_model.eval()
            with torch.no_grad():
                print("Processing Gallery........")
                gallery_feature = extract_feature(global_model, test_dataloaders['gallery'], device)
                print("Processing Query..........")
                query_feature = extract_feature(global_model, test_dataloaders['query'], device)
            result = {"gallery_f": gallery_feature.cpu().numpy(),
            "gallery_label":np.array(gallery_label), 
            "gallery_cam": np.array(gallery_cam), 
            "query_f": query_feature.cpu().numpy(),
            "query_label":np.array(query_label), 
            "query_cam": np.array(query_cam)}
            query_feature = torch.FloatTensor(result['query_f'])
            query_cam = result['query_cam']
            query_label = result['query_label']
            gallery_feature = torch.FloatTensor(result['gallery_f'])
            gallery_cam = result['gallery_cam']
            gallery_label = result['gallery_label']

            query_feature = query_feature.to(device)
            gallery_feature = gallery_feature.to(device)
            CMC = torch.IntTensor(len(gallery_label)).zero_()
            ap = 0.0
            #print(query_label)
            print("Calculating............")
            for i in tqdm(range(len(query_label))):
                #print(query_feature[i],query_label[i],query_cam[i])
                ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
                if CMC_tmp[0]==-1:
                    continue
                CMC = CMC + CMC_tmp
                ap += ap_tmp
                #print(i, CMC_tmp[0])

            CMC = CMC.float()
            CMC = CMC/len(query_label) #average CMC
            mAP = ap/len(query_label)
            eval_epoch_list.append(epoch+1)
            eval_r1_list.append(CMC[0])
            eval_map_list.append(mAP)
            print('Rank@1:%f \nRank@5:%f \nRank@10:%f \nmAP:%f'%(CMC[0],CMC[4],CMC[9],mAP))
        
        global_model.load_state_dict(bckup_global_state_dict) # load the backup state dict of global model saved earlier
        '''
        Save the checkpoint of specific models of all client and global model of the last epoch
        '''
        if epoch == 99:
            for dataset in dataset_list:
                if scenario == 'ska':
                    specific_weight = {k:v for k, v in local_model[dataset].specific_model.state_dict().items() if k in global_model.state_dict()} # choose only feature attraction layer
                else:
                    specific_weight = {k:v for k, v in local_model[dataset].generalized_model.state_dict().items() if k in global_model.state_dict()}
                checkpoint_path = checkpoint_dir + "/" + f"{dataset}_specific.ckpt"
                torch.save(specific_weight, checkpoint_path)
            checkpoint_path = checkpoint_dir + "/" + "global.ckpt"
            torch.save(global_model.state_dict(), checkpoint_path)
        '''
        Selective Update
        '''
        for dataset in dataset_list:
            local_model[dataset].selective_update(global_model.state_dict())
        '''
        Update Optimizer
        '''
        for scheduler in scheduler_list:
            scheduler.step()
        '''
        Plot the figure
        '''
        global_train_acc_list.append(train_acc)
        global_train_loss_list.append(loss)
        global_valid_acc_list.append(valid_acc)
        global_valid_loss_list.append(valid_loss)
        epoch_list.append(epoch+1)
        axs[0].plot(epoch_list, global_train_acc_list)
        axs[0].plot(epoch_list, global_valid_acc_list)
        axs[1].plot(eval_epoch_list, eval_r1_list)
        axs[1].plot(eval_epoch_list, eval_map_list)
        plt.savefig(checkpoint_dir + "/" + str(epoch+1) + ".png")
        #print("Saving Global Model At " + checkpoint_path)



