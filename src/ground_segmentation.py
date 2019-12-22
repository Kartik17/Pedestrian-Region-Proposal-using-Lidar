import numpy as np
import copy

def ground_segmentation(pc,iter_cycle = 5, threshold = 0.3):
    
    pc = np.hstack((pc,np.arange(pc.shape[0],dtype=int).reshape(-1,1)))
    pc_orig = copy.deepcopy(pc)
    
    pc = valid_region(pc,{'x':[0,60],'z':[-3,0]})
    h_col = np.argmin(np.var(pc[:,:3], axis=0))

    #bins, z_range = np.histogram(pc[:,h_col],20)
    approx_coord = np.median(pc,axis = 0)#z_range[np.argmax(bins)]
    for n in range(iter_cycle):
        cov_mat = np.cov(pc[:,:3].T)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        
        normal_vector = eig_vec[np.argmin(eig_val)]
        height = np.dot(pc[:,:3],normal_vector)-np.dot(approx_coord[:3],normal_vector)
        threshold_mask = np.abs(height) < threshold
        pc = pc[threshold_mask]

    world_mask = np.invert(np.in1d(pc_orig[:,3],pc[:,3]))
    world_points = pc_orig[:,:3][world_mask]
    ground_points = pc[:,:3]

    return ground_points.astype('float32'),world_points.astype('float32')

def valid_region(pc,constraint):

    mask = ((pc[:,2] >= constraint['z'][0]) & (pc[:,2] <= constraint['z'][1])) &\
    	   ((pc[:,0] >= constraint['x'][0]) & (pc[:,0] <= constraint['x'][1]))

    valid_world_points = pc[mask]
    return valid_world_points