import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import json 
import os 
import struct 
import open3d as o3d 
from pyquaternion import Quaternion  
from tqdm import tqdm 
from model.core.voxel.voxel_generator import VoxelGenerator 






class Nuscene():
    def __init__(self, root):
        self.root = root 
        self.table_names = ['attribute', 'calibrated_sensor', 'category', 'ego_pose', 
                          'instance', 'log', 'map', 'sample', 'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility']
        self.attribute = self.read_table('attribute.json')
        self.calibrated_sensor = self.read_table('calibrated_sensor.json')
        self.category = self.read_table('category.json')
        self.ego_pose = self.read_table('ego_pose.json')
        self.instance = self.read_table('instance.json')
        self.log = self.read_table('log.json')
        self.map = self.read_table('map.json')
        self.sample = self.read_table('sample.json')
        self.sample_annotation = self.read_table('sample_annotation.json')
        self.sample_data = self.read_table('sample_data.json')
        self.scene = self.read_table('scene.json')
        self.sensor = self.read_table('sensor.json')
        self.visibility = self.read_table('visibility.json')

        self.token2ind = self.token2ind()
        self.sample_decorate() 





    def read_table(self, table_name):
        path = os.path.join(self.root, table_name)
        f = open(path, 'r')
        file = f.read() 
        table = json.loads(file)
        return table 
    

    def token2ind(self):
        token2ind = {}
        for i in range(len(self.table_names)):
            token2ind[self.table_names[i]] = {}

        for i in range(len(self.attribute)):
            token2ind['attribute'][self.attribute[i]['token']] = i 

        for i in range(len(self.calibrated_sensor)):
            token2ind['calibrated_sensor'][self.calibrated_sensor[i]['token']] = i 

        for i in range(len(self.category)):
            token2ind['category'][self.category[i]['token']] = i 

        for i in range(len(self.ego_pose)):
            token2ind['ego_pose'][self.ego_pose[i]['token']] = i 

        for i in range(len(self.instance)):
            token2ind['instance'][self.instance[i]['token']] = i 

        for i in range(len(self.log)):
            token2ind['log'][self.log[i]['token']] = i 

        for i in range(len(self.map)):
            token2ind['map'][self.map[i]['token']] = i 

        for i in range(len(self.sample)):
            token2ind['sample'][self.sample[i]['token']] = i 

        for i in range(len(self.sample_annotation)):
            token2ind['sample_annotation'][self.sample_annotation[i]['token']] = i 

        for i in range(len(self.sample_data)):
            token2ind['sample_data'][self.sample_data[i]['token']] = i 

        for i in range(len(self.scene)):
            token2ind['scene'][self.scene[i]['token']] = i 

        for i in range(len(self.sensor)):
            token2ind['sensor'][self.sensor[i]['token']] = i 

        for i in range(len(self.visibility)):
            token2ind['visibility'][self.visibility[i]['token']] = i 

        
        return token2ind 
    
    
    def get(self, table_name, token):
        
        if table_name == 'attribute':
            return self.attribute[self.token2ind['attribute'][token]]
        
        if table_name == 'calibrated_sensor':
            return self.calibrated_sensor[self.token2ind['calibrated_sensor'][token]]
        
        if table_name == 'category':
            return self.category[self.token2ind['category'][token]]
        
        if table_name == 'ego_pose':
            return self.ego_pose[self.token2ind['ego_pose'][token]]
        
        if table_name == 'instance':
            return self.instance[self.token2ind['instance'][token]]
        
        if table_name == 'log':
            return self.log[self.token2ind['log'][token]]
        
        if table_name == 'map':
            return self.map[self.token2ind['map'][token]]
        
        if table_name == 'sample':
            return self.sample[self.token2ind['sample'][token]]
        
        if table_name == 'sample_annotation':
            return self.sample_annotation[self.token2ind['sample_annotation'][token]]
        
        if table_name == 'sample_data':
            return self.sample_data[self.token2ind['sample_data'][token]]
        
        if table_name == 'scene':
            return self.scene[self.token2ind['scene'][token]]
        
        if table_name == 'sensor':
            return self.sensor[self.token2ind['sensor'][token]]
        
        if table_name == 'visibility':
            return self.visibility[self.token2ind['visibility'][token]]
        


    def sample_decorate(self):

        # Decorate(add short-cut) sample_annotation table with for category_name 
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (add short-cut) sample_data with sensor information
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse index sample with sample_data and annotation 
        for record in self.sample:
            record['data'] = {}
            record['anns'] = [] 

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        
        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        



class Box:
    """  Simple data class representing a 3D box including lable and dimension """
    def __init__(self, center, size, orientation, label):

        assert type(center) == np.ndarray
        assert len(size) == 3
        assert type(orientation) == Quaternion 
        

        self.center = center 
        self.size = size 
        self.orientation = orientation 
        self.label = label 


    def translate(self, x):
        self.center = self.center + x 

    def rotate(self, orientation):
        self.center = np.dot(orientation.rotation_matrix, self.center)
        self.orientation = orientation * self.orientation 






class NusceneData(Dataset):
    def __init__(self, data_path, meta_path, x_min= -15, x_max=15, y_min=-70, y_max =70, z_min=-1.5, z_max=0.5, test=False):

        self.data_path = data_path
        self.meta_path  = meta_path 
         

        
        self.nusc = Nuscene(self.meta_path)
        self.x_min = x_min
        self.x_max = x_max 
        self.y_min = y_min 
        self.y_max = y_max 
        self.z_min = z_min 
        self.z_max = z_max 


        
        # make object of VoxelGenerator Class 
        self.obj = VoxelGenerator([1,1,1], [self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max], 300, 140*30)

        self.files = []
        
        self.mapping = {'animal':'void', 'movable_object.debris':'void', 'movable_object.pushable_pullable':'void', 
                        'static_object.bicycle_rack':'void', 'vehicle.emergency.ambulance':'void', 'vehicle.emergency.police':'void',
                        'movable_object.barrier':'barrier', 'vehicle.bicycle':'bicycle', 'vehicle.bus.bendy':'bus', 'vehicle.bus.rigid':'bus', 
                        'vehicle.car':'car', 'vehicle.construction':'cunstruction_vehicle',
                        'vehicle.motorcycle':'motorcycle', 'human.pedestrian.adult':'pedestrian', 'human.pedestrian.child':'pedestrian', 
                        'human.pedestrian.construction_worker':'pedestrian', 'human.pedestrian.police_officer':'pedestrian', 'human.pedestrian.personal_mobility':'void', 
                        'human.pedestrian.stroller':'void', 'human.pedestrian.wheelchair':'void', 'movable_object.trafficcone':'traffic_cone',
                        'vehicle.trailer':'trailer', 'vehicle.truck':'truck'}
        
        self.class_index = {'void': 0, 'barrier':1, 'bicycle':2, 'bus':3, 'car':4, 'cunstruction_vehicle':5, 'motorcycle':6, 'pedestrian':7, 
                            'traffic_cone':8, 'trailer':9, 'truck':10}

        for i in range(len(self.nusc.scene)):
            my_scene = self.nusc.scene[i] 
            flag = 1 
            first_flag = 1
            while(flag==1):
                if first_flag == 1:
                    first_sample_token = my_scene['first_sample_token']
                    last_sample_token = my_scene['last_sample_token']

                    my_sample = self.nusc.get('sample', first_sample_token)
                    next_sample = my_sample['next']
                    first_flag = 0
                else:
                    my_sample = self.nusc.get('sample', first_sample_token)
                    next_sample = my_sample['next']

                # get sample   LIDAR_TOP data 
                sensor = 'LIDAR_TOP'
                lidar_data = self.nusc.get('sample_data', my_sample['data'][sensor])
                file_name = lidar_data['filename']
                 
                # get pcd file
                file_path = os.path.join(self.data_path, file_name)
                pcd = self.convert_bin_to_pcd(file_path)

                # Retrive sensor and ego pose record 
                cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
                pose_record = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

                # get Annotation data 
                annotation_token_list = my_sample['anns']
                annotation_list = []

                for k in range(len(annotation_token_list)):
                    annotation_token = annotation_token_list[k] 
                    ann_record = self.nusc.get('sample_annotation', annotation_token)

                    category = ann_record['category_name']
                    # map to categegory classes 
                    label = self.mapping[category]

                    
                    center = ann_record['translation']
                    size = ann_record['size']
                    rotation = ann_record['rotation']
                    

                    # create box object 
                    box = Box(center=np.array(center), size=size, orientation=Quaternion(rotation), label=label)

                    # Move box to ego vehicle coordinate system 
                    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                    box.translate(-np.array(pose_record['translation']))
                    box.rotate(Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)]).inverse)

                    # Move box to sensor coordinate system
                    box.translate(-np.array(cs_record['translation']))
                    yaw = Quaternion(cs_record['rotation']).yaw_pitch_roll[0]
                    box.rotate(Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)]).inverse)
                    
                    # Check bounding box is in predefined range or not. If yes then append otherwise discard
                    ann_flag = self.bb_range_check(box.center, box.size)
                    if ann_flag == 1:
                         annotation_list.append([box.center, box.size, box.orientation, box.label])
                    else:
                        continue 


                sample = {}
                sample['pcd'] = pcd 
                sample['annotation_list'] = annotation_list 

                self.files.append(sample)


                first_sample_token = next_sample 

                if next_sample == '':
                    flag = 0

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        pcd = self.files[index]['pcd']
        #print("hello")
        # voxelize the point  cloud 
        voxels, coors, num_point_per_voxel = self.obj.generate(pcd, 140*30)

        annotation_list = self.files[index]['annotation_list']
        
        # make classification logits ternsor for Focal loss calculation shape(num_classes+1, x_max-x_min, y_max-y_min)
        classification_logits = torch.zeros(12, int(self.y_max -self.y_min), int(self.x_max - self.x_min))
        width = int(self.y_max - self.y_min)
        length = int(self.x_max - self.x_min)
        
        center_index = []
        for i in range(len(annotation_list)):
            # get the center coordinate and shift w.r.t. grid 
            x,y,z = annotation_list[i][0]
            # get the class label 
            label = annotation_list[i][3]
            # get class index 
            class_index = self.class_index[label]

            classification_logits[class_index][int(width/2-(y))][int(length/2 + (x))] = 1 
            center_index.append((int(width/2 - (y)), int(length/2+(x))))

        for i in range(int(self.y_max - self.y_min)):
            for j in range(int(self.x_max - self.x_min)):
                if (i, j) not in center_index :
                    classification_logits[11][i][j] = 1 
        
        

        # make detection tensor shape([score, x,y,z, w,l,h, theta], width, length)

        detection_tensor = torch.zeros(8, int(self.y_max - self.y_min), int(self.x_max - self.x_min))
        for i in range(len(annotation_list)):
            # get center coordinates 
            x, y, z = annotation_list[i][0]
            # get width, length, height 
            w, l, h = annotation_list[i][1]
            # get orientation in radians 
            theta = annotation_list[i][2].radians 

            detection_tensor[0][int(width/2 -(y))][int(length/2 +(x))] = 1 
            detection_tensor[1][int(width/2 - (y))][int(length/2 +(x))] = x 
            detection_tensor[2][int(width/2 - (y))][int(length/2  + (x))] = y
            detection_tensor[3][int(width/2  -(y))] [int(length/2  + (x))] =  z 
            detection_tensor[4][int(width/2  - (y))][int(length/2  + (x))] = w 
            detection_tensor[5][int(width/2  -(y))][int(length/2  +(x))] = l 
            detection_tensor[6][int(width/2  -(y))][int(length/2  + (x))] = h 
            detection_tensor[7][int(width/2  - (y))][int(length/2  +(x))] = theta 



        return {'voxels':voxels, 'classification_logits':classification_logits, 'detection_tensor':detection_tensor}
    
    

    
    
    def bb_range_check(self, center, size):
        """ this function check that bounding box is in predefined range of not. If yes return 1 otherwise return 0. """
        x,y,z = center 
        w, l, h = size  


        # get the coordinates of box 
        coordinates = np.array([[l/2, w/2, h/2],
                                [-l/2, w/2, h/2],
                                [-l/2, -w/2, h/2],
                                [l/2, -w/2, h/2],
                                [l/2, w/2, -h/2],
                                [-l/2, w/2, -h/2],
                                [-l/2, -w/2, -h/2],
                                [l/2, -w/2, -h/2]])
        

        coordinates = coordinates + center 


        x_coordinates = coordinates[:, 0]
        y_coordinates = coordinates[:, 1]

        x_coord_min = min(x_coordinates)
        x_coord_max = max(x_coordinates)
        y_coord_min = min(y_coordinates)
        y_coord_max = max(y_coordinates)

        if self.x_min <= x_coord_min and x_coord_max <= self.x_max and self.y_min <= y_coord_min and y_coord_max <= self.y_max :
            return 1 
        else :
            return 0
        


    def convert_bin_to_pcd(self, file_path):
        if file_path.endswith('bin'):
            size_float = 4
            list_pcd = []
            with open(file_path, 'rb') as f:
                byte = f.read(size_float * 5)
                while byte:
                    x,y,z,intensity,ring_index = struct.unpack('fffff', byte)
                    list_pcd.append([x,y,z])
                    byte = f.read(size_float * 5)
            
            pcd = np.asarray(list_pcd)

            return pcd 
        
        
        





        


        


