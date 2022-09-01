from torch import nn
import torchvision.models as models  

class Global_model(nn.Module):
    def __init__(self):
        super(Global_model, self).__init__()
        self.feature_attraction_layer = models.resnet50(pretrained = True)
        self.feature_attraction_layer.fc = nn.Identity()
    def forward(self, x):
        x = self.feature_attraction_layer(x)
        return x

class Client_model(nn.Module):
    def __init__(self, out_feature):
        super(Client_model, self).__init__()
        self.out_feature = out_feature
        self.feature_attraction_layer = models.resnet50(pretrained = True)
        self.feature_attraction_layer.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024,self.out_feature)
        )

    def forward(self, x):
        x = self.feature_attraction_layer(x)
        x = self.classifier(x)
        return x

