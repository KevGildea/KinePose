## [Auto.exe](Dist/Auto.exe)

This tool automates human pose annotation in videos using the YOLOv8 model, designed for 2D pose estimation on the MS COCO dataset. Run on a Windows machine as administrator.

### Key features:

- **Set Start and End Time**: Define the video segment to be processed by specifying the start and end times (in seconds).
- **Open Video**: Load a video file for pose estimation and annotation.
- **Automatic Pose Detection**: Detects and annotates human poses using the YOLOv8 pose estimator.
- **Keypoint Connections**: Visualise pose connections for keypoints.
- **Progress Tracking**: Displays a progress bar while processing video frames.
- **Save Annotated Frames**: Saves the annotated frames and keypoint data in a designated output folder.

### Outputs:
- **Annotated Frames**: Saves each frame as both a raw and annotated image.
- **Keypoint Data**: Saves the detected keypoints for each frame in a text file (`.txt`) with the coordinates of key body parts.

### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/Auto.PNG" alt="Auto.exe" width="300">
  <br>
  <i>The GUI of the Auto.exe application.</i>
</p>


## [Semi-auto.exe](Dist/Semi-auto.exe)

This tool assists with semi-automatic human pose annotation in images using the YOLOv8 model for keypoint detection. Users can refine the automatically detected keypoints by dragging them to the correct locations. Run on a Windows machine as administrator. Video demo available here: [VideoDemo_x175.mp4](Demo/VideoDemo_x175.mp4). 

### Key features:

- **Open Image**: Load an image for pose annotation.
- **Automatic Keypoint Detection**: Automatically detects human pose keypoints using the YOLOv8 pose estimator.
- **Manual Keypoint Adjustment**: Allows users to click and drag keypoints to correct their position.
- **Save Annotations**: Save the adjusted keypoints to the corresponding `.txt` file.
- **Save Annotated Images**: Save the image with visualised keypoints and skeletal connections.

### Outputs:
- **Annotated Keypoints**: Saves the manually adjusted keypoints for each body part to the corresponding `.txt` file.
- **Annotated Images**: Saves two versions of the image, one with keypoints and another with skeletal connections.

### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/Semi-auto.PNG" alt="Semi-auto.exe" width="300">
  <br>
  <i>The GUI of the Semi-auto.exe application.</i>
</p>


## [Manual.exe](Dist/Manual.exe)

This tool allows for fully manual annotation of human pose keypoints in images. Users manually select each keypoint. Run on a Windows machine as administrator.

### Key features:

- **Open Image**: Load an image for manual pose annotation.
- **Manual Keypoint Annotation**: Users click to place keypoints on the image.
- **Keypoint Visualisation**: Displays each keypoint as it is annotated, with labels showing the body part.
- **Undo Annotation**: Right-click to undo the last annotated keypoint.
- **Save Annotations**: Save the manually annotated keypoints to a corresponding `.txt` file.
- **Save Annotated Images**: Save the image with visualised keypoints and skeletal connections.

### Outputs:
- **Annotated Keypoints**: Saves the manually selected keypoints for each body part to a corresponding `.txt` file.
- **Annotated Images**: Saves two versions of the image, one with keypoints and skeletal connections and another with just skeletal connections.

### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/Manual.PNG" alt="Manual.exe" width="300">
  <br>
  <i>The GUI of the Manual.exe application.</i>
</p>



## MS COCO pose format

<img src="../images/MSCOCO.png" alt="MS COCO pose format" width="100"/>


## To Do:
1. Outut confidence scores
2. Custom pose tool
