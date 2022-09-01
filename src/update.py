import torch
from torch import nn

from tqdm.auto import tqdm
import random
import re
from torch.hub import load_state_dict_from_url
import models_ska as resnets
#import resnet_vallina as resnets

class LocalUpdate(object):
    def __init__(self, device, criterion, dataloaders, output_dim, local_epoch, total_global_epoch):
        self.device = device
        self.criterion = criterion
        self.dataloaders = dataloaders
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth') # Fedavg
        checkpoint = torch.load("model_best.pth.tar", map_location = 'cpu')
        st_dict = {k[15:]: v for k, v in checkpoint['state_dict'].items()}

        self.specific_model = resnets.__dict__['resnet50'](client=True, out_feature=output_dim)
        self.specific_model.load_state_dict(st_dict, strict=False)
        self.generalized_model = resnets.__dict__['resnet50'](client=True, out_feature=output_dim)
        self.generalized_model.load_state_dict(state_dict, strict=False)
        self.generalized_model.load_state_dict(self.specific_model.state_dict())
        for p1, p2 in zip(self.specific_model.parameters(), self.generalized_model.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                print("they are not the same")
        self.local_aggregated_model = resnets.__dict__['resnet50']()
        
        self.local_epoch = local_epoch
        self.total_global_epoch = total_global_epoch

        self.specific_optimizer = torch.optim.SGD([{"params":self.specific_model.fc.parameters(), "lr":0.1}, 
                                                    {"params":self.specific_model.conv1.parameters(), "lr":0.01},
                                                    {"params":self.specific_model.bn1.parameters(), "lr":0.01},
                                                    {"params":self.specific_model.layer1.parameters(), "lr":0.01},
                                                    {"params":self.specific_model.layer2.parameters(), "lr":0.01},
                                                    {"params":self.specific_model.layer3.parameters(), "lr":0.01},
                                                    {"params":self.specific_model.layer4.parameters(), "lr":0.01},], 
                                                lr = 0.01, momentum = 0.9, weight_decay = 5e-4, nesterov = True)

        self.generalized_optimizer = torch.optim.SGD([{"params":self.generalized_model.fc.parameters(), "lr":0.1}, 
                                                    {"params":self.generalized_model.conv1.parameters(), "lr":0.01},
                                                    {"params":self.generalized_model.bn1.parameters(), "lr":0.01},
                                                    {"params":self.generalized_model.layer1.parameters(), "lr":0.01},
                                                    {"params":self.generalized_model.layer2.parameters(), "lr":0.01},
                                                    {"params":self.generalized_model.layer3.parameters(), "lr":0.01},
                                                    {"params":self.generalized_model.layer4.parameters(), "lr":0.01},], 
                                                lr = 0.01, momentum = 0.9, weight_decay = 5e-4, nesterov = True)

    def update_weights(self):
        #self.specific_model.to(self.device)
        self.generalized_model.to(self.device)

        #self.specific_model.train()
        self.generalized_model.train()

        epoch_loss = []

        for epoch in range(self.local_epoch):
            '''
            Training Phase
            '''
            #self.specific_model.train()
            self.generalized_model.train()

            train_accs = []
            train_loss = []

            for data in tqdm(self.dataloaders['train']):
                inputs, labels = data
                
                #specific_output = self.specific_model(inputs.to(self.device))
                generalized_output = self.generalized_model(inputs.to(self.device))

                #specific_loss = self.criterion(specific_output, labels.to(self.device))
                generalized_loss = self.criterion(generalized_output, labels.to(self.device))

                #specific_loss.backward()
                generalized_loss.backward()
                #(specific_loss + generalized_loss).backward()

                #self.specific_optimizer.step()
                self.generalized_optimizer.step()

                #self.specific_optimizer.zero_grad()
                self.generalized_optimizer.zero_grad()

            #acc = ((specific_output.argmax(dim=-1)) == labels.to(self.device)).float().mean()
            #train_loss.append(specific_loss.item())
            acc = ((generalized_output.argmax(dim=-1)) == labels.to(self.device)).float().mean()
            train_loss.append(generalized_loss.item())
            train_accs.append(acc)
            #print(train_loss)
            #print(train_accs)

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)

            print(f"[ Train | {epoch + 1:03d}/{self.local_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")    

            epoch_loss.append(train_loss)
            '''
            Validation Phase
            '''
            loss = 0.0
            total = 0.0
            correct = 0.0
            #self.specific_model.eval()
            self.generalized_model.eval()
            for data in tqdm(self.dataloaders['val']):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #outputs = self.specific_model(inputs)
                outputs = self.generalized_model(inputs)
                batch_loss = self.criterion(outputs, labels)
                #print(batch_loss)
                loss += batch_loss.item()
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            valid_acc = correct/total

            print(f"[ Valid | {epoch + 1:03d}/{self.local_epoch:03d} ] loss = {loss:.5f}, acc = {valid_acc:.5f}")
        #return self.specific_model.state_dict(), self.generalized_model.state_dict() , sum(epoch_loss) / len(epoch_loss), loss, train_acc, valid_acc
        return self.generalized_model.state_dict() , sum(epoch_loss) / len(epoch_loss), loss, train_acc, valid_acc
    def selective_update(self, global_weights):
        self.generalized_model.load_state_dict(global_weights, strict = False)
        #layers_for_specific = {k: v for k, v in global_weights.items() if not re.findall("bn",k)}
        #self.specific_model.load_state_dict(layers_for_specific, strict = False)    
    