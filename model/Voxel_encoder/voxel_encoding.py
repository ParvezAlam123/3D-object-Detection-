import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 20)


    def forward(self, x):
        batch, num_voxel, num_points, dim = x.shape 

        query = key = value = x 

        score = (query @ key.transpose(2,3))  / np.sqrt(dim)

        attention = F.softmax(score, -1)

        context = attention @ value

        output = F.relu(self.fc1(context))

        # max in dimension of num_points (output: B * num_voxel * dim)

        output = torch.max(output, dim=2)[0]

        
        return output 




       














    





