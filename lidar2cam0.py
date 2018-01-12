import cv2
import numpy as np
import pykitti
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Change this to the directory where you store KITTI data
basedir = '/home/vitob/git_repo/kitti_dataset_explore'

# Specify the dataset to load
date = '2011_09_26'
drive = '0017'

frame_range = range(0, 3, 1)

# Load the data. Optionally, specify the frame range to load.
# Passing imformat='cv2' will convert images to uint8 and BGR for
# easy use with OpenCV.
dataset = pykitti.raw(basedir, date, drive,
                      frames=frame_range, imformat='cv2')

# velodyne to cam0 matrix, shape = 4x4 = (R | t)
T_v2c0 = dataset.calib.T_cam0_velo
# cam0 intrinsics matrix, shape = 3x3
K_c0 = dataset.calib.K_cam0

# Insert a column of 0 - cam0 has zero translation
# For other cameras, it will not be zero column.
K_c0 = np.append(K_c0, np.zeros((3,1)), axis=1)
# Shape 3x4 = matmul(3x4, 4x4)
P_v2c0 = np.matmul(K_c0, T_v2c0)

# Load data from next synchronized frame - KITIT data seem to have 1:1 lidar:cam data rate
cam0_generator = dataset.cam0
velo_generator = dataset.velo

for i in frame_range:
    cam0 = next(cam0_generator)
    velo = next(velo_generator)
    # Reduce number of points - pick one every 5 rows
    velo = velo[::5,:]
    
    # Rule out those invisible points
    velo_visible = []
    for i in range(velo.shape[0]):
        if velo[i,0]>5:
            velo_visible.append(velo[i,:])
    velo_v = np.array(velo_visible)
    
    # Project velo data into image (cam0) coordinates
    tmp = np.matmul(P_v2c0, np.c_[velo_v[:,0:3],np.ones(velo_v.shape[0]).transpose()].transpose()) 
    tmp = tmp.transpose()
    # Normalize to (x,y) coordinates within image
    tmp2 = tmp[:,:-1]/tmp[:,2].reshape((tmp.shape[0],1))
    
    # Grab depth info. 
    #   BUG WAS: velo_v[:,3] - reflective strength, velo_v[:,0] - x depth
    #   Velodyne's coordinates are different from camera
    depth = velo_v[:,0]
    
    # Show the gray img from camera 0
    #plt.imshow(cam0, cmap='gray', interpolation='nearest', aspect='auto')
    plt.imshow(cam0, cmap='gray', interpolation='nearest')
    
    visible_dots = []
    # Draw dots from lidar onto this image
    for i in range(depth.shape[0]):
        # Filter out those dots that are not in FOV
        if tmp2[i][0] >0 and tmp2[i][0]<cam0.shape[1] and tmp2[i][1]>0 and tmp2[i][1]<cam0.shape[0]:
            visible_dots.append([tmp2[i][0], tmp2[i][1], depth[i]])
    
    visible_dots = np.reshape(visible_dots, (len(visible_dots),3))
    
    # Draw the lidar depth onto cam0 image with color mapping 
    plt.scatter(x=visible_dots[:,0], y=visible_dots[:,1], c=visible_dots[:,2], cmap='viridis_r', marker='+')
    plt.show()
    
    # Use opencv to show image
    cv2.imshow('cam0',cam0)
    k = cv2.waitKey(-1)
    cv2.destroyAllWindows()
