## [wMP.exe](Dist/wMP.exe)

This tool performs 3D human pose reconstruction using the **weighted midpoint triangulation** method with camera calibration and 2D keypoint data from two different camera views. It allows users to visualise 3D poses using a graphical interface. Run on a Windows machine as administrator.

### Key features:

- **Load Calibration and Keypoints**: Load camera calibration files and folders containing 2D keypoints for triangulation.
- **Weighted Midpoint Triangulation**: Uses weighted midpoint triangulation to estimate 3D poses from two camera views, with optional confidence weights for improved accuracy.
- **COCO-to-H36M Mapping**: Maps keypoints from MS COCO format to the Human3.6M (H36M) format for pose reconstruction.
- **Pose Visualisation**: Provides an interactive 3D visualisation of the reconstructed pose, with sliders to explore different frames and animated pose playback.
- **Savitzky-Golay Smoothing**: Smooths the pose data over time using the Savitzky-Golay filter to reduce noise.
- **Save 3D Keypoints**: Outputs the 3D pose keypoints in a text file for further processing.

### Outputs:
- **3D Pose Data**: Saves the 3D keypoints for each frame in a `.txt` file.


### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/wMP1.PNG" alt="wMP.exe" width="250">
  <br>
  <i>The GUI of the wMP.exe application.</i>
</p>


<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/wMP2.PNG" alt="wMP.exe" width="500">
  <br>
  <i>Triangulated pose visualisation in the wMP.exe application.</i>
</p>


## To Do:
1. Implement weights
2. Custom pose tool
3. Add an option to refine 2D poses through reprojecting each 3D triangulated keypoint back to each image. The viewer should include images from each view with functionality of clicking and dragging keypoint positions, and a visulisation of the resulting changes to the 3D pose.
