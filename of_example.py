import numpy as np
import cv2
import math

cap = cv2.VideoCapture('cam0_01.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.3,
                       minDistance = 7,
                        blockSize = 7 )
 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#p0 = cv2.goodFeaturesToTrack(old_gray,50,0.01,10)
#tmp = np.int0(p0)
#for i in tmp:
#    x,y = i.ravel()
#    cv2.circle(old_gray, (x,y), 2, [0,255,0],-1)
#cv2.imshow('corners',old_gray)
#k = cv2.waitKey(-1)
#cv2.destroyAllWindows()

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

for j in range(60):
#while(1):
    ret,frame = cap.read()
    if ret==False:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Filter out points with big errors
    #st[err>1.0]=0
    # If too few points, exit
    #if np.count_nonzero(st[:,0])<10:
    #    break

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # distance list
    dist = []
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 1)
        frame = cv2.circle(frame,(a,b),3,color[i].tolist(),-1)
        dist_tmp = math.sqrt((a-c)*(a-c) + (b-d)*(b-d))
        dist.append(dist_tmp)
    img = cv2.add(frame,mask)
    print(np.mean(dist), np.std(dist))

    cv2.imshow('frame',img)
#    k = cv2.waitKey(-1)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc key pressed
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

k = cv2.waitKey(-1)
cv2.destroyAllWindows()
cap.release()
