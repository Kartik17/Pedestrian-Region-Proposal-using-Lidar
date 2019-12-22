import numpy as np
from ground_segmentation import *
from lidar_to_img import *
import cv2
from utils import *
from sklearn.cluster import DBSCAN

lidar_path = "./Data/2011_09_28/velodyne_points/data/"
#lidar_path = "./Data/velodyne/training/velodyne/"

def filter_cluster(pc_obj_list):
	pc_obj_array = np.array(pc_obj_list)
	
	mask = (((np.max(pc_obj_array,axis = 0)-np.min(pc_obj_array,axis = 0)) < np.array([2,2,3])) \
		& ((np.max(pc_obj_array,axis = 0)-np.min(pc_obj_array,axis = 0)) > np.array([0.2,0.2,0.2]))) 
	
	if np.all(mask):
		return True
	else:
		return False

def clustering(pc):
	pc_obj = pcl.PointCloud()
	pc_obj.from_array(pc[:,:3])
	clustering_people = DBSCAN(eps=0.17, min_samples=10,leaf_size=30).fit(pc)

	core_samples_mask_people = np.zeros_like(clustering_people.labels_, dtype=bool)
	core_samples_mask_people[clustering_people.core_sample_indices_] = True
	labels_people = clustering_people.labels_

	clusters_list = []
	for index in set(labels_people):
		temp = pc[labels_people == index]
		if filter_cluster(temp):
			clusters_list.append(temp)
	
	return clusters_list
	
