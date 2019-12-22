# Utilities Script 
import pcl
import numpy as np
from ground_segmentation import *
import pcl.pcl_visualization
import struct
from lidar_to_img import *
import cv2
import os


def lidar_to_image(cluster,calib_path):

	lidar_to_cam = get_lidar_cam_calib(calib_path)
	cam_dict = get_cam_to_cam_calib(calib_path)

	# Cartesian to Homogenous
	points = cart_to_hom(np.array(cluster))
	
	# Velodyne to Camera Reference frame
	cam_points = np.dot(lidar_to_cam,points.T).T
	
	'''
	if distortion:
		cam_rect_x = cam_points[:,0]/cam_points[:,2]
		cam_rect_y = cam_points[:,1]/cam_points[:,2]

		#Account for Distortion
		r = cam_rect_x**2 + cam_rect_y**2
		cam_rect_x_dist = cam_rect_x*(1+dc[0]*(r) + dc[1]*(r**2) + dc[4]*(r**3)) + 2*dc[2]*cam_rect_x*cam_rect_y +dc[3]*(r +2*(cam_rect_x**2))
		cam_rect_y_dist = cam_rect_y*(1+dc[0]*(r) + dc[1]*(r**2) + dc[4]*(r**3)) + dc[2]*(r +2*(cam_rect_y**2)) + 2*dc[3]*cam_rect_x*cam_rect_y 

		cam_points = np.array([cam_rect_x_dist,cam_rect_y_dist,np.ones_like(cam_rect_y_dist),np.ones_like(cam_rect_y_dist)]).T
	'''

	proj_points = np.dot(cam_dict['rect_proj_matrix'],cam_points.T).T
	proj_2dpoints = proj_points[:,0:2]/proj_points[:,2].reshape(-1,1)

	return proj_2dpoints

def bbox_from_clusters(clusters,ground_z,calib_path):
	n = 0
	bbox_list = []

	for cluster in clusters:
		if True:
			cluster = np.vstack((cluster,np.array([np.mean(cluster[:,0]),np.mean(cluster[:,1]),ground_z])))
			proj_2dpoints = lidar_to_image(cluster,calib_path)

			start_point = (np.int(np.min(proj_2dpoints[:,0]))-5,np.int(np.max(proj_2dpoints[:,1])))
			end_point = (np.int(np.max(proj_2dpoints[:,0])),np.int(np.min(proj_2dpoints[:,1]))-10)

			bbox_list.append((start_point,end_point))

	return bbox_list

def visualize_bbox_image(bbox_list,imagepath,idx):

	image = cv2.imread(os.path.join(imagepath,str(idx) + '.png'))
	for bbox in bbox_list:
		try:
			print(bbox[0],bbox[1])
			cv2.rectangle(image, bbox[0], bbox[1], (255,0,0), 2)
		except:
			pass
	cv2.imshow('image',image)
	cv2.waitKey(0)

def visualize_pc_points_image(clusters,calib_path,imagepath,idx):

	image = cv2.imread(os.path.join(imagepath,str(idx) + '.png'))
	for cluster in clusters:
		proj_2dpoints = lidar_to_image(cluster,calib_path)
		try:
			for i in np.arange(proj_2dpoints.shape[0]):
				cv2.circle(image,(int(proj_2dpoints[i,0]),int(proj_2dpoints[i,1])),1,(0,0,255),-1)
		except:
			pass
	cv2.imshow('image',image)
	cv2.waitKey(0)


def visualize_clusters(clusters):
	cluster_point_list = []
	for cluster in clusters:
		cluster_point_list.extend(cluster)
	cluster_cloud = pcl.PointCloud() 
	cluster_cloud.from_list(cluster_point_list)
	
	viewer = pcl.pcl_visualization.PCLVisualizering()
	
	while  True:
		viewer.AddPointCloud(cluster_cloud, b'scene_cloud', 0)
		viewer.SpinOnce()

def visualize_pc(pc):
	pc_pcl = pcl.PointCloud()
	pc_pcl.from_array(pc)
	viewer = pcl.pcl_visualization.PCLVisualizering()
	
	while  True:
		viewer.AddPointCloud(pc_pcl, b'scene_cloud', 0)
		viewer.SpinOnce()

def valid_pc_points(pc,imagepath,calib_path,idx):

	
	image = cv2.imread(os.path.join(imagepath,str(idx) + '.png'))
	img_shape = image.shape

	pts_img = lidar_to_image(pc,calib_path)

	val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
	val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
	val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
	pts_valid_flag = np.logical_and(val_flag_merge, pc[:,0] >= 0)


	return pc[pts_valid_flag]

def save_proposals(bbox_list,idx):
	with open('./results/bbox/' + str(idx) + '.txt','w+') as f:
		for bbox in bbox_list:
			f.write(str(bbox[0]) +","+str(bbox[1])+'\n')
	f.close()

def save_bbox_image(bbox_list,imagepath,idx):
	image = cv2.imread(os.path.join(imagepath,str(idx) + '.png'))
	for bbox in bbox_list:
		try:
			cv2.rectangle(image, bbox[0], bbox[1], (255,0,0), 2)
		except:
			pass
	cv2.imwrite('./results/bbox_images/'+ str(idx)+'.png',image)