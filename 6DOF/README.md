## [KinePose.exe](Dist/KinePose.exe)

This tool performs 6DOF inverse kinematics (IK) analysis on human pose data using 3D keypoints and a user-defined kinematic chain. It includes options for both frame-by-frame and temporal optimisations, with features to load and process kinematic chain data, keypoints, and motion visualisations. Run on a Windows machine as administrator

### Key features:

- **Load Kinematic Chain**: Load the kinematic chain (in Excel format) for the human body model (HBM), defining joint orientations and positions, and joint ranges of motion (ROMs).
- **Load 3D Keypoints**: Load 3D pose keypoints (in `.txt` format).
- **Inverse Kinematics**: Perform frame-based or temporal IK optimisation to align the kinematic chain with the loaded 3D pose keypoints.
- **Motion Smoothing**: Apply the Savitzky-Golay filter to smooth IK results across frames.
- **Visualisation**: Visualise both the kinematic chain and the pose data, with options to display animations using a slider.
- **Save Results**: Save the transformation data (local/global orientations, positions, and angular velocities) to Excel files for further analysis.

Video demo available here: [VideoDemo_x175.mp4](Demo/VideoDemo_x175.mp4).

### Outputs:
- **Transformation Data**: Saves local and global joint orientations and positions in `.xlsx` files.
- **Angular Velocities**: Saves angular velocities for each joint in `.xlsx` format.

### GUI

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose1.PNG" alt="KinePose.exe" width="400">
  <br>
  <i>The GUI of the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose2.PNG" alt="KinePose.exe" width="400">
  <br>
  <i>Visualisation of the user-defined kinematic chain in the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose3.PNG" alt="KinePose.exe" width="500">
  <br>
  <i>Visualisation of the 3D pose in the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose4.PNG" alt="KinePose.exe" width="400">
  <br>
  <i>Visualisation of pose loss vectors and keypoint indices, and optional setting of initial guess for optimisation in the KinePose.exe application.</i>
</p>

<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/KinePose5.PNG" alt="KinePose.exe" width="500">
  <br>
  <i>Visualisation of the optimised 6DOF pose in the KinePose.exe application.</i>
</p>


<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/chains.png" alt="KinePose chains" width="700">
  <br>
  <i>Examples of the variety of kinematic chain configurations that may be specified.</i>
</p>


## To Do:
1. Document the formatting of user inputs, and outputs.
