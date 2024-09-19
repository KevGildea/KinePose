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



## Semi-auto.exe

This tool assists with semi-automatic human pose annotation in images using the YOLOv8 model for keypoint detection. Users can refine the automatically detected keypoints by dragging them to the correct locations. Key features:

- **Open Image**: Load an image for pose annotation.
- **Automatic Keypoint Detection**: Automatically detects human pose keypoints using the YOLOv8 pose estimator.
- **Manual Keypoint Adjustment**: Allows users to click and drag keypoints to correct their position.
- **Keypoint Visualization**: Keypoints and their connections are displayed for clarity (e.g., nose, eyes, shoulders, elbows, etc.).
- **Save Annotations**: Save the adjusted keypoints to a `.txt` file.
- **Save Annotated Images**: Save the image with visualized keypoints and skeletal connections.

### Outputs:
- **Annotated Keypoints**: Saves the manually adjusted keypoints for each body part to a `.txt` file.
- **Annotated Images**: Saves two versions of the image, one with keypoints and another with skeletal connections, in `.png` format.






## To Do:
1. Outut confidence scores
2. Custom pose tool
