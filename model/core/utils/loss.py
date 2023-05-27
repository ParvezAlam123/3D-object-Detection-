import numpy as np 
import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 





def box_loss(box_p, box_g):
    score_p = box_p[0]
    x_p = box_p[1]
    y_p = box_p[2]
    z_p = box_p[3] 
    w_p = box_p[4]
    l_p = box_p[5]
    h_p = box_p[6]
    theta_p = box_p[7]   
   
    
    score_g = box_g[0]
    x_g = box_g[1]
    y_g = box_g[2]
    z_g = box_g[3]   
    w_g = box_g[4] 
    l_g = box_g[5] 
    h_g = box_g[6] 
    theta_g = box_g[7]

    

    if score_g != 0 :

       delta_x = abs((x_g - x_p) / l_g)
       delta_y = abs((y_g - y_p) / w_g)
       delta_z = abs((z_g - z_p) / h_g)

       delta_w = abs(torch.log(w_g / w_p))
       delta_l = abs(torch.log(l_g / l_p))
       delta_h = abs(torch.log(h_g / h_p))

       delta_theta = abs(torch.sin(theta_g - theta_p))

       delta_score = abs(score_p - score_g)


       total_loss = delta_x + delta_y + delta_z + delta_w + delta_l + delta_h + delta_score 
    
    else:
        delta_score = abs(score_p - score_g)
        total_loss = delta_score 

    return total_loss 







def focal_loss(input, target, gamma):
    ce = F.cross_entropy(input, target, reduction='none')
    pt = torch.exp(-ce)

    focal_loss = (1 - pt)**gamma * ce 

    # take sum of focal loss of all classes 
    focal_loss = focal_loss.sum()

    return focal_loss 
    
  









    



