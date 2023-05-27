import torch 
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F 



from model.core.voxel.voxel_generator import  VoxelGenerator 
from model.backbone.backbone_3d import Backbone, Features 
from model.head.head_network import MultiHeadAttention, ClassificationHead, DetectionHead 




class Network(nn.Module):
    """  
    Args:
       voxel_size(List[float]): size of a single voxel.
       point_cloud_range (List(Float)):  range of the points. [x_min, y_min, z_min, x_max, y_max, z_max].
       max_num_points(int): Maximum number of the points in a single voxel.
       max_voxels (int, optional): Maximum number of the voxels.
       d_model (int): dimension of feature of every layer after applying MLP, (512)
    """
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, d_model, num_classes):
        super().__init__() 

        self.voxel_size = voxel_size 
        self.point_cloud_range = point_cloud_range 
        self.max_num_points = max_num_points 
        self.max_voxels = max_voxels 
        self.d_model = d_model 
        self.num_classes = num_classes
        self.x_min, self.y_min, self.z_min , self.x_max, self.y_max, self.z_max = self.point_cloud_range 




        #self.voxel_generator = VoxelGenerator(voxel_size=self.voxel_size, 
                           # point_cloud_range=self.point_cloud_range, 
                            #max_num_points=self.max_num_points,max_voxels=self.max_voxels)

        # convert dim 3 to 12 
        self.fc = nn.Linear(3, 12)

        self.backbone = Backbone()
        self.features = Features()
        self.multiheadattention = MultiHeadAttention(d_model=self.d_model)
        self.classificationhead = ClassificationHead(x_min=self.x_min, y_min=self.y_min,
                                                    x_max=self.x_max, y_max=self.y_max,
                                                     input_dim=self.d_model,num_classes=self.num_classes)
        self.detectionhead = DetectionHead(x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, 
                                          y_max=self.y_max, input_dim=self.d_model)

    

    def forward(self, voxels):
        # get voxel of point clouds (num_voxel, max_point, dim)

        #voxels, coors, num_points_per_voxel = self.voxel_generator.generate(pcd, self.max_voxels)

        # get voxel feature using self-attention mechanism
        query=key=value=voxels 
        batch, n_voxel, num_points, dim = voxels.shape 
        scale = np.sqrt(dim)
        attention_weight = torch.matmul(query,key.transpose(2,3)) / scale 
        attention_weight = F.softmax(attention_weight, dim=-1)
        feature = torch.matmul(attention_weight, value)

        feature = F.relu(self.fc(feature))
        

        # give voxel to backbone network 
        feature_level1, feature_level2, feature_level3, feature_level4 = self.backbone(feature)

        # convert features into same dimension 

        feature_level1, feature_level2, feature_level3, feature_level4 = self.features(feature_level1, feature_level2, feature_level3, feature_level4)
        
        #change shape into [B, 1, 512]
        feature_level1 = torch.unsqueeze(feature_level1, dim=1)
        feature_level2 = torch.unsqueeze(feature_level2, dim=1)
        feature_level3 = torch.unsqueeze(feature_level3, dim=1)
        feature_level4 = torch.unsqueeze(feature_level4, dim=1)

        # concatenate features (batch, num_level)
        features = torch.cat((feature_level1, feature_level2, feature_level3, feature_level4), dim=1)
        

        # apply multiheaded attention block 

        feature = self.multiheadattention(features)

        # change the shape of fetures
        B,s,_ = feature.shape 
        feature = feature.reshape(B, -1)

        # apply classification head 

        classification_output = self.classificationhead(feature)

        # apply detection head 

        detection_output = self.detectionhead(feature)

        return detection_output, classification_output 
    










                                          












        
