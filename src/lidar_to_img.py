import numpy as np
import cv2
import os

calib_path = './Data/2011_09_28/calibration/'

def get_lidar_cam_calib(calib_path):
	V2C = np.eye(4)
	calib_path = os.path.join(calib_path,'calib_velo_to_cam.txt') 
	file = open(calib_path)
	f = file.readlines()
	for line in f:
		(identifier,data) = line.split(':',1)
		if identifier =='R':
			V2C[:3,:3] = np.fromstring(data, sep=' ').reshape(3,3)
		elif identifier == 'T':
			V2C[:3,-1] = np.fromstring(data, sep=' ')

	return V2C

def get_cam_to_cam_calib(calib_path):
	cam_dict = {}

	calib_path = os.path.join(calib_path,'calib_cam_to_cam.txt') 
	file = open(calib_path)
	data = file.readlines()
	cam_data = data[18:26]

	cam_dict['rect_proj_matrix'] =  np.array([np.array(cam_data[7].split()[1:4]+['0'],dtype = 'float32'),
							   			   	  np.array(cam_data[7].split()[5:8]+['0'],dtype = 'float32'),
							   			      np.array(cam_data[7].split()[9:12]+['0'],dtype = 'float32'),
							   			   	  np.array([0,0,0,1])])
	return  cam_dict

def cart_to_hom(pts):
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

if __name__ == '__main__':
	print(get_lidar_cam_calib(calib_path))
	cam_dict = get_cam_to_cam_calid(calib_path)
	print(cam_dict['rect_proj_matrix'])



