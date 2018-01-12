# kitti_dataset_explore
Getting familiar with Kitti dataset.


[img1]: http://www.cvlibs.net/datasets/kitti/images/setup_top_view.png

KITTI setup:
![alt text][img1] 

Sample data from KITTI dataset. Too large thus not checked in. Devkit/readme.txt has details on all data here.
```
2011_09_26
├── 2011_09_26_drive_0017_extract
│   ├── image_00
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
...
    │   └── timestamps.txt
│   ├── image_01
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   └── timestamps.txt
...
│   ├── image_02
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
...
│   │   └── timestamps.txt
│   ├── image_03
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
...
│   │   └── timestamps.txt
│   ├── oxts
│   │   ├── data
│   │   │   ├── 0000000000.txt
│   │   │   ├── 0000000001.txt
...
│   │   ├── dataformat.txt
│   │   └── timestamps.txt
│   └── velodyne_points
│       ├── data
│       │   ├── 0000000000.txt
│       │   ├── 0000000001.txt
...
│       ├── timestamps.txt
│       ├── timestamps_end.txt
│       └── timestamps_start.txt
├── 2011_09_26_drive_0017_sync
│   ├── image_00
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
...
│   │   └── timestamps.txt
│   ├── image_01
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
...
│   │   └── timestamps.txt
│   ├── image_02
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
...
│   │   └── timestamps.txt
│   ├── image_03
│   │   ├── data
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
...
│   │   └── timestamps.txt
│   ├── oxts
│   │   ├── data
│   │   │   ├── 0000000000.txt
│   │   │   ├── 0000000001.txt
...
│   │   ├── dataformat.txt
│   │   └── timestamps.txt
│   ├── tracklet_labels.xml
│   └── velodyne_points
│       ├── data
│       │   ├── 0000000000.bin
│       │   ├── 0000000001.bin
...
│       ├── timestamps.txt
│       ├── timestamps_end.txt
│       └── timestamps_start.txt
├── calib_cam_to_cam.txt
├── calib_imu_to_velo.txt
└── calib_velo_to_cam.txt
```

