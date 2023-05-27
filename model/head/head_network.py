import torch
import torch.nn as nn
import torch.nn.functional  as F
import numpy as np 



class AttentionHead(nn.Module):
    def __init__(self, d_head):
        super().__init__() 

        self.d_head = d_head 

        self.fc1 = nn.Linear(self.d_head, 2 * self.d_head)
        self.fc2 = nn.Linear(2 * self.d_head, self.d_head)

    def forward(self, x):
        B, num_level, d_head = x.shape 

        scale = np.sqrt(2 * d_head)

        query = self.fc1(x)
        key = F.relu(self.fc1(x))
        value = self.fc1(x)

        attention_weight = torch.bmm(query, key.transpose(1,2)) / scale 
        attention_weight = F.softmax(attention_weight, dim=-1)


        feature = torch.bmm(attention_weight, value)

        feature = F.relu(self.fc2(feature))

        return feature 





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()

        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads 

        self.head1 = AttentionHead(self.d_head)
        self.head2 = AttentionHead(self.d_head)
        self.head3 = AttentionHead(self.d_head)
        self.head4 = AttentionHead(self.d_head)
        self.head5 = AttentionHead(self.d_head)
        self.head6 = AttentionHead(self.d_head)
        self.head7 = AttentionHead(self.d_head)
        self.head8 = AttentionHead(self.d_head)

        

        self.batchnorm = nn.BatchNorm1d(num_features=4)  # 4 is the number of level in backbone
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        """ x is in shape 1, Batch*num_level, dim"""
        B, num_level, dim = x.shape
        # reshape dimension 
        x = x.reshape(self.n_heads, B, num_level, self.d_head)

        head_feature1 = self.head1(x[0])
        head_feature2 = self.head2(x[1])
        head_feature3 = self.head3(x[2])
        head_feature4 = self.head4(x[3])
        head_feature5 = self.head5(x[4])
        head_feature6 = self.head6(x[5])
        head_feature7 = self.head7(x[6])
        head_feature8 = self.head8(x[7])


        feature = torch.cat((head_feature1, head_feature2, 
                            head_feature3, head_feature4, head_feature5, head_feature6,head_feature7, head_feature8), dim=-1)

        
        x = x.reshape(B, num_level, dim) # shape into original dimension

        # add batch norm and residal connection
        feature = x + feature
        feature = self.batchnorm(feature)

        # add mlp
        output = F.relu(self.fc(feature))

        # add residual connection 

        output = feature + output 
        output = self.batchnorm(output)

        return output 




class ClassificationHead(nn.Module):
    def __init__(self,x_min, y_min, x_max, y_max, input_dim, num_classes=11):   # front is x and left right is y axis in this 
        super().__init__() 

        self.width = x_max - x_min 
        self.length = y_max - y_min 
        self.num_classes = num_classes 
        self.input_dim = input_dim 

        self.fc1 = nn.Linear(self.input_dim*4, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, (self.num_classes) * self.width * self.length )  


    def forward(self, x):
        B, _ = x.shape 

        

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        out = out.reshape(B, -1, self.length, self.width)  # [Batch, num_classes+1, l, w]


        return out 




class DetectionHead(nn.Module):
    def __init__(self, x_min, y_min, x_max, y_max, input_dim):
        super().__init__() 

        self.length = x_max - x_min 
        self.width = y_max - y_min 
        self.input_dim = input_dim 

        self.fc1 = nn.Linear(self.input_dim*4, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, (1+3+3+1) * self.width * self.length)  # [(score, delta_x,delta_y,delta_z,
                                                                          #  delta_l, delta_w,delta_h, delta_theta) * grid_length * grid_width]


    def forward(self, x):
        B, _ = x.shape 

        

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        out = out.reshape(B, -1, self.width, self.length)    # (Batch, 8 , grid_l, grid_w)


        return out 




















        






        

