import torchvision.models as models
import torch.nn as nn
import torch

def resnet18():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    return model

def resnet18_binary():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    return model

if __name__ == '__main__':
    model = resnet18()
    x = torch.Tensor(64, 3, 512, 512)
    print(x.shape)
    x = model(x)
    x = nn.Sigmoid()(x)
    print(x.shape)
    #print(model)
