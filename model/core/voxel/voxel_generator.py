import torch 
import torch.nn as nn
import numpy as np 


class VoxelGenerator():
    """
    Args: 
    vexel_size (list[float]): size of each voxel [x,y,z]
    point_cloud_range(list[float]): size of point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    max_num_points: maximum number of points in each voxel
    max_voxels : maximum number of voxels      
    """
    def __init__(self,voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size 
        grid_size = np.round(grid_size).astype(np.int64)


        self._voxel_size = voxel_size 
        self._point_cloud_range = point_cloud_range 
        self._max_num_points = max_num_points 
        self._max_voxels = max_voxels 
        self._grid_size = grid_size 

    def generate(self, points, max_voxels):
        return points_to_voxel(points, self._voxel_size, self._point_cloud_range, self._max_num_points,  max_voxels)

    
    @property 
    def voxel_size(self):
        return self._voxel_size 

    @property 
    def max_num_points_per_voxel(self):
        return self._max_num_points 

    @property 
    def point_cloud_range(self):
        return self._point_cloud_range 

    @property 
    def grid_size(self):
        return self._grid_size 

    

def points_to_voxel(points, voxel_size, coors_range, max_points=35,  max_voxels=20000):
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.type)
    
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size 
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    num_points_per_voxel = np.zeros((max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(voxelmap_shape, dtype=np.int32)

    voxels = np.zeros((max_voxels, max_points, points.shape[-1]), dtype=np.int32)
    coors = np.zeros((max_voxels, 3),dtype=np.int32)

    voxel_num, coors, voxels, num_points_per_voxel = points_to_voxel_kernel(points, voxel_size, coors_range, num_points_per_voxel,
                                coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    #coors = coors[:voxel_num]
    #voxels = voxels[:voxel_num]
    #num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel 



def points_to_voxel_kernel(points, 
                          voxel_size,
                          coors_range,
                          num_points_per_voxel,
                          coor_to_voxelidx,
                          voxels, 
                          coors,
                          max_points=35,
                          max_voxels=20000):
    
    N = points.shape[0]
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size 
    grid_size = np.round(grid_size).astype(np.int64)


    coor = np.zeros((3, ), dtype=np.int32)
    voxel_num = 0
    failed = False 
    for i in range(N):
        failed = False 
        for j in range(ndim):
            c = np.floor((points[i][j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True 
                break 
            coor[j] = c 
        if failed:
            continue 
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1 :
            voxelidx = voxel_num 
            if voxel_num >= max_voxels:
                break 
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx 
            coors[voxelidx] = coor 

        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1 

    return voxel_num , coors,voxels,  num_points_per_voxel 






