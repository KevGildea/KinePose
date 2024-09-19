Compiled onefile .exe's available here (Run as administrator): 

1. [Auto.exe](Dist/Auto.exe)
2. [Semi-auto.exe](Dist/Semi-auto.exe)
2. [Manual.exe](Dist/Manual.exe)

Video demo available here: [VideoDemo_x175.mp4](Demo/VideoDemo_x175.mp4)



## Auto.exe

This tool automates human pose annotation in videos using the YOLOv8 model, designed for 2D pose estimation on the MS COCO dataset. Key features:

- **Set Start and End Time**: Define the video segment to be processed by specifying the start and end times (in seconds).
- **Open Video**: Load a video file for pose estimation and annotation.
- **Automatic Pose Detection**: Detects and annotates human poses using the YOLOv8 pose estimator.
- **Keypoint Connections**: Visualize pose connections for keypoints such as nose, eyes, shoulders, elbows, wrists, hips, knees, and ankles.
- **Progress Tracking**: Displays a progress bar while processing video frames.
- **Save Annotated Frames**: Saves the annotated frames and keypoint data in a designated output folder.

### Outputs:
- **Annotated Frames**: Saves each frame as both a raw and annotated image.
- **Keypoint Data**: Saves the detected keypoints for each frame in a text file (`.txt`) with the coordinates of key body parts.





## To Do:
1. Outut confidence scores
2. Custom pose tool
