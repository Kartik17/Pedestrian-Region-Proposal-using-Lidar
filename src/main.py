from ground_segmentation import *
from lidar_to_img import *
from utils import *
import os
from clustering import *
import time

data_path = './Data/2011_09_28'

def load_velodyne(lidar_path,idx):
	path = os.path.join(lidar_path,'data',str(idx)+'.bin')
	pc = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:,:3]
	
	return pc

def dataloader(data_path):
	calib_path = os.path.join(data_path,'calibration')
	velodyne_path = os.path.join(data_path,'velodyne_points')
	image_path = os.path.join(data_path,'image_02')

	idx_list = [f.split('.')[0] for f in os.listdir(os.path.join(velodyne_path,'data'))]
	return idx_list,calib_path,velodyne_path,image_path



if __name__ == '__main__':
	

	idx_list,calib_path,velodyne_path,image_path = dataloader(data_path)

	start_time = time.time()
	counter = 0
	for idx in idx_list:
		pc = load_velodyne(velodyne_path,idx)
		pc_valid = valid_pc_points(pc,os.path.join(image_path,'data'),calib_path,str(idx))
		pc_ground, pc_world = ground_segmentation(pc_valid)
		ground = np.median(pc_ground[:,2])
		clusters = clustering(pc_world)
		#visualize_pc(pc)
		#visualize_pc(pc_valid)
		#visualize_pc(pc_world)
		#exit()
		#visualize_clusters(clusters)
		#exit()
		#print(len(clusters))
		bbox_list = bbox_from_clusters(clusters,ground,calib_path)
		counter += len(bbox_list)
		#save_proposals(bbox_list,idx)
		#save_bbox_image(bbox_list,os.path.join(image_path,'data'),idx)
		#visualize_pc_points_image(clusters,calib_path, os.path.join(image_path,'data'), str(idx))
		visualize_bbox_image(bbox_list,os.path.join(image_path,'data'), str(idx))
	print('Time:{}\n'.format(time.time() - start_time))
	print('Counter:{}\n'.format(counter))