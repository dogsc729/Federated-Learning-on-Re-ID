import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np

from tqdm.auto import tqdm
import random

config = {
    "global_iteration": 100,
    "local_optimisation_epoch": 1,
    "lr_feature_extraction": 0.01,
    "lr_classifier": 0.1,
    "batch_size": 32,
    "warm_epoch": 5,
}

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

'''
Transform
'''
train_transform_list = [
    transforms.Resize((256,128)),
    transforms.ToTensor()
]

train_transform = transforms.Compose(train_transform_list)


'''
Dataloader
'''

#dataset_list = ['MSMT17','cuhk03-np-detected','DukeMTMC-reID', 'Market', ]
dataset_list = ['DukeMTMC-reID']
for dataset in dataset_list:
    data_dir = f'datasets/{dataset}/pytorch'

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=train_transform)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = config['batch_size'], 
                shuffle = True, num_workers = 2, pin_memory = True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("***************")
    print(dataset)
    print("train size:", dataset_sizes['train'])
    print("val size:", dataset_sizes['val'])
    #print(len(image_datasets['train'].classes)) #how many classes are there in training set.

print("Dataloader set up success")

'''
Model
'''
if torch.cuda.is_available():
    print("Using GPU")
    device = 'cuda'
else:
    print("Using CPU")
    device = 'cpu'

class Global_model(nn.Module):
    def __init__(self):
        super(Global_model, self).__init__()
        self.feature_attaction_layer = models.resnet50(pretrained = True)

    def forward(self, x):
        x = self.feature_attaction_layer(x)
        return x

class Market_model(nn.Module):
    def __init__(self):
        super(Market_model, self).__init__()
        self.feature_attaction_layer = models.resnet50(pretrained = True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000,751)
        )

    def forward(self, x):
        x = self.feature_attaction_layer(x)
        x = self.classifier(x)
        return x

class MSMT_model(nn.Module):
    def __init__(self):
        super(MSMT_model, self).__init__()
        self.feature_attaction_layer = models.resnet50(pretrained = True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000,1041)
        )

    def forward(self, x):
        x = self.feature_attaction_layer(x)
        x = self.classifier(x)
        return x

class Duke_model(nn.Module):
    def __init__(self):
        super(Duke_model, self).__init__()
        self.feature_attaction_layer = models.resnet50(pretrained = False)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000,702)
        )

    def forward(self, x):
        x = self.feature_attaction_layer(x)
        x = self.classifier(x)
        return x        

class cuhk03_np_model(nn.Module):
    def __init__(self):
        super(cuhk03_np_model, self).__init__()
        self.feature_attaction_layer = models.resnet50(pretrained = True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000,767)
        )

    def forward(self, x):
        x = self.feature_attaction_layer(x)
        x = self.classifier(x)
        return x    

model = Duke_model().to(device)

global_model = Global_model().to(device)
'''
Train
'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 5e-4, nesterov = True)


n_epochs = 1
model.train()
#print(model)
for epoch in range(n_epochs):

    train_loss = []
    train_accs = []
    for data in tqdm(dataloaders['train']):
        inputs, labels = data
        n,c,h,w = inputs.shape
        output = model(inputs.to(device))
        #print(output.size())
        #print(labels.size())

        loss = criterion(output, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=-1)) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()
    valid_loss = []
    valid_accs = []

    for data in dataloaders['val']:
        input, labels = data
        with torch.no_grad():
            output = model(input.to(device))

        loss = criterion(output, labels.to(device))
        acc = ((output.argmax(dim=-1)) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


