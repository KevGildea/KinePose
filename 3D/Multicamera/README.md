Compiled onefile .exe available here: [wMP.exe](Dist/wMP.exe) (Run as administrator).

## wMP.exe

This tool performs 3D human pose reconstruction using the **weighted midpoint triangulation** method with camera calibration and 2D keypoint data from two different camera views. It allows users to visualize and refine 3D poses using a graphical interface. Key features:

- **Load Calibration and Keypoints**: Load camera calibration files and folders containing 2D keypoints for triangulation.
- **Weighted Midpoint Triangulation**: Uses weighted midpoint triangulation to estimate 3D poses from two camera views, with optional confidence weights for improved accuracy.
- **COCO-to-H36M Mapping**: Maps keypoints from COCO format to the Human3.6M (H36M) format for pose reconstruction.
- **Pose Visualization**: Provides an interactive 3D visualization of the reconstructed pose, with sliders to explore different frames and animated pose playback.
- **Savitzky-Golay Smoothing**: Smooths the pose data over time using the Savitzky-Golay filter to reduce noise.
- **Save 3D Keypoints**: Outputs the 3D pose keypoints in a text file for further processing.

### Outputs:
- **3D Pose Data**: Saves the 3D keypoints for each frame in a `.txt` file.
- **3D Pose Visualization**: Displays the reconstructed 3D pose in real-time with the option to save the animation.



## To Do:
1. Implement weights
2. Custom pose tool
