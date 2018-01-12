import cv2
import numpy as np
import scipy 
import scipy.linalg 
import pykitti
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Local scripts
import tile_of_func as of


#-------------
# Global Parameters
#-------------
## Interactive mode - show image of every frame interactively.
INTERACTIVE = False

# No of frames to process
frame_range = range(0, 50, 1)

# Dataset location
# Change this to the directory where you store KITTI data
basedir = '/work/git_repo/kitti_dataset_explore'

# Specify the dataset to load
date = '2011_09_26'
drive = '0001'


# Tile size
tile_size = 100





#-------------
# Main(): Processing cam, lidar data
#-------------

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

for fr in frame_range:
    #Pyplot figure size
    plt.figure(figsize=(12,4))
    plt.autoscale(tight=True)

    cam0 = next(cam0_generator)
    velo = next(velo_generator)

    # Annotate lidar onto cam0 image
    of.lidar_proc(cam0, velo, T_v2c0, K_c0, P_v2c0)

    # Optical flow tracking
    # Feature list for consistency check
    feature_list = []

    # First frame, mark feature
    if fr==0:

        (cam0_height, cam0_width) = cam0.shape
        print("image size (w, h): ", cam0_width, cam0_height)

        # Split image into tile_size to find features
        p0 = of.OF_TileFindFeature(cam0, tile_size, of.feature_params)
        #DEBUG print(p0.shape, p0)
        plt.scatter(p0[:,:,0], p0[:,:,1], marker='o', color='b')

    # Optical flow tracking
    else:


        print("Frame ", fr, ":")
        # Forward OF tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_cam0, cam0, p0, None, **of.lk_params)

        good_old = p0[st==1]
        good_new = p1[st==1]

        # Backward OF tracking
        p2, st2, err = cv2.calcOpticalFlowPyrLK(cam0, old_cam0, p1, None, **of.lk_params)

        # Clean-up feature_list
        feature_list = of.OF_FeatureListFilter(p0, p1, p2, cam0_width, cam0_height)

#        # Draw filtered feature points and associated OF
#        for idx in range(feature_list.shape[0]):
#            plt.plot([feature_list[idx,0], feature_list[idx,2]], 
#                            [feature_list[idx,1], feature_list[idx,3]],
#                            linestyle='-', color='plum')

#        # Draw old feature points
#        #plt.scatter(good_old[:,0], good_old[:,1], marker='^', c='g')
#        # Draw new feature points
#        #plt.scatter(good_new[:,0], good_new[:,1], marker='o', c='r')
#        # Draw optical flow
#        for idx in range(good_old.shape[0]):
#            plt.plot([good_old[idx,0], good_new[idx,0]], 
#                            [good_old[idx,1], good_new[idx,1]],
#                            linestyle='-', color='b')

        # Find affine parameters sx,sy,tx,ty for each grid
        of.OF_TileAffineSolver(cam0, tile_size, feature_list)

        # Save points for next round
        p0 = good_new.reshape(-1,1,2)



    old_cam0 = cam0.copy()

    # Save pyplot image
    plt.savefig(str(fr).zfill(3)+'_of.png')
    # Show the OF tracking image
    if INTERACTIVE:
        plt.show()
    
    # Use opencv to show image
    #cv2.imshow('cam0',cam0)
    #k = cv2.waitKey(-1)
    #cv2.destroyAllWindows()
