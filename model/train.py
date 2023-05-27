import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from model.Dataset.nuscene import NusceneData 
from torch.utils.data import DataLoader 
from model.complete_net.network import Network 
from model.core.utils.loss import focal_loss 
from model.core.utils.loss import  box_loss 
import time 



# set device 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters 
data_path_train ="/home/parvez_alam/Data/Nuscene_Asia/Train/v1.0-trainval01_blobs"
meta_path_train = "/home/parvez_alam/Data/Nuscene_Asia/Train/Metadata/v1.0-trainval_meta/v1.0-trainval"
data_path_test = "/home/parvez_alam/Data/Nuscene_Asia/Test/v1.0-test_blobs"
meta_path_test = "/home/parvez_alam/Data/Nuscene_Asia/Test/Metadata/v1.0-test_meta/v1.0-test"
x_min = -15
x_max = 15
y_min = -70
y_max = 70
z_min = -3
z_max = 1 
epochs = 100                # number of epoch for training

voxel_size = np.array([1,1,1])
point_cloud_range = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
max_num_points = 300
max_voxels = 20000
d_model = 512 
num_classes = 11+1


# Creates dataloader
#train_ds = NusceneData(data_path=data_path_train, meta_path=meta_path_train, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
valid_ds = NusceneData(data_path=data_path_test, meta_path=meta_path_test, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, test=True)

#train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=1, shuffle=True)

print("length = ", len(valid_ds))







# creates model 

model = Network(voxel_size=voxel_size, point_cloud_range=point_cloud_range,
                 max_num_points=max_num_points, max_voxels=max_voxels, d_model=d_model, num_classes=num_classes)

# send model to device 
model.to(device)

# load checkpoint and trained model
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch_number"] 

model.load_state_dict(loaded_checkpoint["model_state"])
model.eval() 



# optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


training_loss = []   # list for storing loss per epoch

def train(model, train_loader,  epochs, lambda1, lambda2):

    for i in range(epochs):
        running_loss = 0.0
        

        for n, data in enumerate(train_loader):
            voxels = data['voxels'].to(device).float()
            classification_logits = data['classification_logits'].to(device).float()
            detection_tensor = data['detection_tensor'].to(device).float() 

                # send pcd to network 

            detection_output, classification_output = model(voxels)

                # calculate detection loss
            loss = 0.0 
            _, C, W, L = detection_output.shape 
            

            
            for k in range(W):
                for j in range(L):
                    score_p = detection_output[:,0,k,j]
                    x_p = detection_output[:, 1,k,j]
                    y_p = detection_output[:, 2,k,j]
                    z_p = detection_output[:, 3,k,j]
                    w_p = detection_output[:, 4,k,j]
                    l_p = detection_output[:, 5,k,j]
                    h_p = detection_output[:, 6,k,j]
                    theta_p = detection_output[:, 7,k,j]


                    box_p = [score_p, x_p, y_p, z_p, w_p, l_p, h_p, theta_p]

                    score_g = detection_tensor[:, 0,k,j]
                    x_g = detection_tensor[:, 1,k,j]
                    y_g = detection_tensor[:, 2,k,j]
                    z_g = detection_tensor[:, 3,k,j]
                    w_g = detection_tensor[:, 4,k,j]
                    l_g = detection_tensor[:, 5,k,j]
                    h_g = detection_tensor[:, 6,k,j]
                    theta_g = detection_tensor[:, 7,k,j]

                    box_g = [score_g, x_g, y_g, z_g, w_g, l_g, h_g, theta_g]

                    detection_loss =  box_loss(box_p, box_g)

                        

                    classification_loss = focal_loss(classification_output, classification_logits,2)
                    
                    # Multi task loss 
                    if detection_loss == detection_loss  and  classification_loss == classification_loss:
                       loss = loss + lambda2 * detection_loss + lambda1  * classification_loss 

                    #detection_loss.detach()
                    #classification_loss.detach()
                    
                
 
            #detection_loss = torch.tensor([loss], requires_grad=True)

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step() 

            running_loss = running_loss + loss.item()

            del loss 
            del detection_output
            del classification_output 


            print("n = {}, iteration = {} ".format(n+1, i+1))
            
            
            


        print("Number of epoch = {}, running_loss = {}".format(i+1, running_loss))
        training_loss.append(running_loss)
        



        # save checkpoint
        checkpoint = {
            "epoch_number": i+1,
            "model_state" : model.state_dict()
        }
        torch.save(checkpoint, "checkpoint.pth")






def valid(model, valid_loader):
    #calculation of AP for every class 
    TP = np.zeros((14, 12))
    FP = np.zeros((14, 12))
    FN = np.zeros((14, 12))

    for n , data in enumerate(valid_loader):
           voxels = data['voxels'].to(device).float()
           classification_logits = data['classification_logits'].to(device).float() 
           detection_tensor = data['detection_tensor'].to(device).float() 
        
           detection_output , classification_output = model(voxels)


           # appli softmax on classification_output 
           classification_output = F.softmax(classification_output, dim=1)
           max_prob_index = torch.max(classification_output, dim=1)[1]



        
           B, C, W, H = detection_tensor.shape 
        
        
           # define threshold values 
           threshold = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
           for b in range(B):
              for k in range(W):
                  for l in range(H):
                      for i in range(len(threshold)):
                          score = abs(detection_output[b][0][k][l].item()) + 0.5
                          if score >= threshold[i]:
                              class_index = max_prob_index[b][k][l].item()
                              if classification_logits[b][class_index][k][l].item() == 1:
                                   TP[i][class_index] = TP[i][class_index] + 1 
                              else:
                                   FP[i][class_index] = FP[i][class_index] + 1 
                          else:
                               class_index = max_prob_index[b][k][l]
                               FN[i][class_index] = FN[i][class_index] + 1
        
        
        

     
    print("TP= ", TP)
    print("FP = ", FP)
    print("FN= ", FN)
    
    precision = TP / ((TP + FP) + 0.000001)
    recall = TP / ((TP + FN)  + 0.000001)
        
    print("precision =", precision)
    print("recall= ", recall)
    AP = np.zeros(12)

    for k in range(len(AP)):
        sum = 0
        for j in range(13):
            sum = sum + (recall[j+1][k] - recall[j][k]) * precision[j+1][k]
        
        AP[k] = sum 

        
    print("AP=", AP)
    # calculattion of mAP 
    sum = 0 
    for i in range(len(AP)):
        sum = sum + AP[i]
    
    mAP = sum / 12 





    print("AP of Void = ", AP[0])
    print("AP of barrier = ", AP[1])
    print("AP of bicycle = ", AP[2])
    print("AP of bus = ", AP[3])
    print("AP of car = ", AP[4])
    print("AP of construction_vehicle= ", AP[5])
    print("AP of motor_cycle = ", AP[6])
    print("AP of pedestrian = ", AP[7])
    print("AP of traffic_cone = ", AP[8])
    print("AP of trailer = ", AP[9])
    print("AP of truck = ", AP[10])
    print("AP of background class = ", AP[11])

    print(" mAP = ", mAP)

 

    
         










#train(model=model, train_loader=train_loader, epochs=10, lambda1=1.0, lambda2=2.0)

valid(model, valid_loader)














                    



















            


            








