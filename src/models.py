import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.layers_to_align = ['layer1', 'layer2']

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x
    
class MLP_CIFAR(nn.Module):
    def __init__(self, input_size=3072, hidden_size=512, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, num_classes)

        # Come nel primo esperimento, allineiamo i due layer nascosti
        self.layers_to_align = ['layer1', 'layer2']

    def forward(self, x):
        x = x.view(x.size(0), -1) # Appiattisce l'immagine
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x