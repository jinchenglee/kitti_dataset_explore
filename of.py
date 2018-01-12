import cv2
import numpy as np
import pykitti
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Change this to the directory where you store KITTI data
basedir = '/work/git_repo/kitti_dataset_explore'

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
    # Read image as gray
    cam0 = next(cam0_generator)

    # Color version of this b&w image. 
    # Convert gray to BGR
    cam0_bgr = cv2.cvtColor(cam0, cv2.COLOR_GRAY2BGR)
    
#    # ---------------
#    # Harris Corner
#    # ---------------
#    gray = np.float32(cam0)
#    dst = cv2.cornerHarris(gray,2,3,0.04)
#    #result is dilated for marking the corners, not important
#    dst = cv2.dilate(dst,None)
#    # Mark corners 
#    cam0_bgr[dst>0.01*dst.max()]=[0,0,255]
#    
#    cv2.imshow('Harris Corner',cam0_bgr)
#    k = cv2.waitKey(-1)
#    cv2.destroyAllWindows()
    
    # ----------------
    # Shi-Tomasi Corner
    # ----------------
    
    corners = cv2.goodFeaturesToTrack(cam0,100,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(cam0_bgr,(x,y),2,[0,255,0],-1)
    
    cv2.imshow('Shi-Tomasi',cam0_bgr)
    k = cv2.waitKey(-1)
    cv2.destroyAllWindows()
    
#    # ----------------
#    # Fast Corner
#    # ----------------
#    # Initiate FAST object with default values
#    fast = cv2.FastFeatureDetector_create()
#    
#    # find and draw the keypoints
#    #fast.setNonmaxSuppression(0)
#    kp = fast.detect(cam0,None)
#    fast_out_img = cam0_bgr
#    cv2.drawKeypoints(cam0_bgr, kp, fast_out_img, color=(255,0,0))
#    
#    cv2.imshow('Fast Corner', fast_out_img)
#    k = cv2.waitKey(-1)
#    cv2.destroyAllWindows()
    
    
    
    #k = cv2.waitKey(0)
    #if k == 27:         # wait for ESC key to exit
    #    cv2.destroyAllWindows()
    #elif k == ord('s'): # wait for 's' key to save and exit
    #    cv2.imwrite('boat.png',img)
    #    cv2.destroyAllWindows()
