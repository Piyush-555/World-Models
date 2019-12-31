import torch.nn as nn

class Controller(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Linear(32 + 256, 3)
        
    def forward(self, x):
        return self.fc(x)