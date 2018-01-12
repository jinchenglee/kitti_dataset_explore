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
drive = '0001'

frame_range = range(0, 50, 1)

# Load the data. Optionally, specify the frame range to load.
# Passing imformat='cv2' will convert images to uint8 and BGR for
# easy use with OpenCV.
dataset = pykitti.raw(basedir, date, drive, frames=frame_range, imformat='cv2')

# Load data from next synchronized frame - KITIT data seem to have 1:1 lidar:cam data rate
cam0_generator = dataset.cam0
velo_generator = dataset.velo
for i in frame_range:
    cam0 = next(cam0_generator)
    cam0_bgr = cv2.cvtColor(cam0, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('cam0/cam0_'+str(i).zfill(3)+'.png', cam0_bgr)
