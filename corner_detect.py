import numpy as np
import cv2

# Read image as gray
cam0 = cv2.imread('boat.jpg',0)

# Color version of this b&w image. 
# Convert gray to BGR
cam0_bgr = cv2.cvtColor(cam0, cv2.COLOR_GRAY2BGR)

# ---------------
# Harris Corner
# ---------------
gray = np.float32(cam0)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Mark corners 
cam0_bgr[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('Harris Corner',cam0_bgr)
k = cv2.waitKey(-1)
cv2.destroyAllWindows()

# ----------------
# Shi-Tomasi Corner
# ----------------

corners = cv2.goodFeaturesToTrack(cam0,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(cam0_bgr,(x,y),2,[0,255,0],-1)

cv2.imshow('Shi-Tomasi',cam0_bgr)
k = cv2.waitKey(-1)
cv2.destroyAllWindows()

# ----------------
# Fast Corner
# ----------------
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
#fast.setNonmaxSuppression(0)
kp = fast.detect(cam0,None)
fast_out_img = cam0_bgr
cv2.drawKeypoints(cam0_bgr, kp, fast_out_img, color=(255,0,0))

cv2.imshow('Fast Corner', fast_out_img)
k = cv2.waitKey(-1)
cv2.destroyAllWindows()



#k = cv2.waitKey(0)
#if k == 27:         # wait for ESC key to exit
#    cv2.destroyAllWindows()
#elif k == ord('s'): # wait for 's' key to save and exit
#    cv2.imwrite('boat.png',img)
#    cv2.destroyAllWindows()
