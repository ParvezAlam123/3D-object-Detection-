import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F






class Feature_MLP(nn.Module):
    def __init__(self, dim):
        super().__init__() 

        self.dim = dim 

        self.fc1 = nn.Linear(in_features=self.dim, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))

        return out 



class Backbone(nn.Module):
    def __init__(self):
        super().__init__() 

        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=3)
        self.batchnorm3d_1 = nn.BatchNorm3d(num_features=10)

        self.conv3d_2 = nn.Conv3d(in_channels=10, out_channels=10, kernel_size=3)
        self.batchnorm3d_2 = nn.BatchNorm3d(num_features=10)
        


        self.conv3d_3 = nn.Conv3d(in_channels=10, out_channels=10, kernel_size=3)
        self.batchnorm3d_3 = nn.BatchNorm3d(num_features=10)

        self.conv3d_4 = nn.Conv3d(in_channels=10, out_channels=10, kernel_size=3)
        self.batchnorm3d_4 = nn.BatchNorm3d(num_features=10)




    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)

        x = self.batchnorm3d_1(F.relu(self.conv3d_1(x)))
        feature_level1 = x 

        x = self.batchnorm3d_2(F.relu(self.conv3d_2(x)))
        feature_level2 = x 

        x = self.batchnorm3d_3(F.relu(self.conv3d_3(x)))
        feature_level3 = x 

        x = self.batchnorm3d_4(F.relu(self.conv3d_4(x)))
        feature_level4 = x 


        return feature_level1, feature_level2, feature_level3, feature_level4 







class Features(nn.Module):
    """ This class initialise the mlp for every level features to convert them into same dimensions.
    Args:
        feature1 (torch.Tensor): features of conv level1 
        feature2 (torch.Tensor): feature of conv level2 
        feature3 (torch.Tensor): feature of conv level3 
        feature4 (torch.Tensor) : feature of conv level4

    Return:
       (torch.Tensor) :  features output of all 4 levels.
    """

    def __init__(self, dim=10):
        super().__init__()
        
        self.dim = dim 

        self.mlp1 = Feature_MLP(self.dim)
        self.mlp2 = Feature_MLP(self.dim)
        self.mlp3 = Feature_MLP(self.dim)
        self.mlp4 = Feature_MLP(self.dim)


    def forward(self, feature_level1, feature_level2, feature_level3, feature_level4):

        B, C , _, _, _ = feature_level1.shape 
        
        
        feature_level1 = torch.reshape(feature_level1, (B, C, -1))
        feature_level2 = torch.reshape(feature_level2, (B, C, -1))
        feature_level3 = torch.reshape(feature_level3, (B, C, -1))
        feature_level4 = torch.reshape(feature_level4, (B, C, -1))
        
        # channel wise maxpooling 
        feature_level1 = torch.max(feature_level1, dim=2)[0]
        feature_level2 = torch.max(feature_level2, dim=2)[0]
        feature_level3 = torch.max(feature_level3, dim=2)[0]
        feature_level4 = torch.max(feature_level4, dim=2)[0]

        # apply mlp 

        feature_level1 = self.mlp1(feature_level1)
        feature_level2 = self.mlp2(feature_level2)
        feature_level3 = self.mlp3(feature_level3)
        feature_level4 = self.mlp4(feature_level4)

        
        return feature_level1, feature_level2, feature_level3, feature_level4 
































        









    




    




        

        

        











        


