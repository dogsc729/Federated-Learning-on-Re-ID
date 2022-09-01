
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import os
import time
from tqdm.auto import tqdm
from models import Global_model, Client_model
#import models_ska as resnets
import resnet_vallina as resnets
import numpy as np

def fliplr(img):
    '''
    Flip Horizontal
    '''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def load_network(network):
    save_path = "/home/b07611033/decentralized_reproduce/checkpoint/2022-06-13-05-18-11-normalresnet50_fedavg_laststride/global.ckpt"
    network.load_state_dict(torch.load(save_path))
    return network

def extract_feature(model, dataloaders, device):
    count = 0
    linear_num = 2048

    for iter, data in enumerate(tqdm(dataloaders)):
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        ff = torch.FloatTensor(n, linear_num).zero_().to(device)

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.to(device))
            outputs = model(input_img)
            ff += outputs

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            if iter == 0:
                features = torch.FloatTensor(len(dataloaders.dataset), ff.shape[1])
            start = iter*128 # batch size = 64
            end = min((iter+1)*128, len(dataloaders.dataset))
            features[start:end, :] = ff
    print("Total Image Processed: ", count)
    return features

# write a dataset, get image, label, cameraID
def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        #print(filename)
        #label = filename[0:4]
        label = filename[6:11] # for viper, prid 
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def evaluate(qf,ql,qc,gf,gl,gc):
    #print(gl)
    #print(ql)
    query = qf.view(-1,1) # flatten
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #print(np.max(score))
    #print(np.argmax(score))
    index = np.argsort(score) # small to large
    index = index[::-1]
    
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    #print(query_index)
    #print(camera_index)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True) # same ID but different camera
    #print(good_index) # problem
    #print(np.max(score[good_index]))
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flpythoatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

if __name__ == "__main__":

    '''
    CUDA
    '''
    if torch.cuda.is_available():
        print("==============> Using GPU")
        device = 'cuda:0'
    else:
        print("==============> Using CPU")
        device = 'cpu'
    

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_dir = "../datasets/prid/pytorch"

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                                shuffle=False, num_workers=8) for x in ['gallery','query']}

    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    #print(gallery_path)
    gallery_cam, gallery_label = get_id(gallery_path)
    #print("Gallery Cam", gallery_cam)
    #print("Gallery Label", gallery_label)
    query_cam, query_label = get_id(query_path)
    #print("Query Cam", query_cam)
    #print("Query Label", query_label)
    print("Test Started========================>")

    global_model = resnets.__dict__['resnet50']()
    global_model = load_network(global_model)
    global_model = global_model.eval()

    if use_gpu:
        global_model = global_model.to(device)

    since = time.time()
    with torch.no_grad():
        print("Processing Gallery........")
        gallery_feature = extract_feature(global_model, dataloaders['gallery'], device)
        print("Processing Query..........")
        query_feature = extract_feature(global_model, dataloaders['query'], device)

    #query_feature, gallery_feature,query_cam, query_label, gallery_cam, gallery_label = torch.load("features.pt")
    #print("gallery_feature_size: ", gallery_feature.size())
    #print("query_feature_size: ", query_feature.size())    
    time_elapsed = time.time() - since

    print("Training Complete In {:.0f}m {:.2f}s".format(time_elapsed // 60, time_elapsed % 60))

    result = {"gallery_f": gallery_feature.cpu().numpy(),
            "gallery_label":np.array(gallery_label), 
            "gallery_cam": np.array(gallery_cam), 
            "query_f": query_feature.cpu().numpy(),
            "query_label":np.array(query_label), 
            "query_cam": np.array(query_cam)
    }
    
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam']
    query_label = result['query_label']
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam']
    gallery_label = result['gallery_label']

    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)

    ''' print(query_feature.size())
    print(len(query_cam))
    print(len(query_label))
    print(gallery_feature.size())
    print(len(gallery_cam))
    print(len(gallery_label)) '''
    #torch.save([query_feature, gallery_feature,query_cam, query_label, gallery_cam, gallery_label], "./features.pt")

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
    print('Rank@1:%f \nRank@5:%f \nRank@10:%f \nmAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
