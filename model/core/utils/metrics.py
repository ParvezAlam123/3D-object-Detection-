import numpy as np 
from model.Dataset import Box 



def iou_3d(box1, box2):
    """ Calculates the intersection over Union(IOU) between two 3D bounding boxes
      defined by their center coordinates(x,y,z), length(l), width(w) and height(h)"""
    
    # calculate coodinates of corners of bounding boxes.
    box1_corners = get_box_corners(box1)
    box2_corners = get_box_corners(box2)
    
    # get intersection and union volume 
    intersection_vol = get_intersection_vol(box1_corners, box2_corners)

    box1_vol = get_box_vol(box1)
    box2_vol = get_box_vol(box2)

    union_vol = box1_vol + box2_vol - intersection_vol 

    iou = intersection_vol / union_vol 

    return iou 




def get_box_vol(box):
    """ Calculates the box volume """

    w, l, h = box.size 

    return w * l * h 





def get_box_corners(box):

    x,y,z = box.center 
    w, l, h = box.size 

    corners = np.array([[l/2, w/2, h/2],
                        [-l/2, w/2, h/2],
                        [-l/2, -w/2, h/2],
                        [l/2, -w/2, h/2],
                        [l/2, w/2, -h/2],
                        [-l/2, w/2, -h/2],
                        [-l/2, -w/2, -h/2],
                        [l/2, -w/2, -h/2]])
    
    corners = np.array([x,y,z]) + corners 

    return corners 

def get_intersection_vol(box1_corners, box2_corners):
    """  Calculates the intersection of volumes of two bounding boxes defined by their coordinates of corners"""

    x_overlap = get_1d_overlap(box1_corners[:,0], box2_corners[:, 0])
    y_overlap = get_1d_overlap(box1_corners[:,1], box2_corners[:,1])
    z_overlap = get_1d_overlap(box1_corners[:,2], box2_corners[:,2])
    
    if x_overlap < 0 or y_overlap < 0 or z_overlap < 0 :
        print("Boxes are not overlapped")

    intersection_vol = x_overlap * y_overlap * z_overlap 

    return intersection_vol 



def get_1d_overlap(x1, x2):
    """ calculates overlap in 1 dimension"""

    x1_min = np.min(x1)
    x1_max = np.max(x1)
    x2_min = np.min(x2)
    x2_max = np.max(x2)

    overlap = min(x1_max, x2_max) - max(x1_min, x2_min)

    return overlap 









