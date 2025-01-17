## [KinePose.exe](Dist/KinePose.exe)

This tool uses inverse kinematics (IK) to map 3D human pose data onto a user-defined kinematic chain. Run on a Windows machine as administrator.

### Key features:

- **Load kinematic chain**: Load the kinematic chain (in Excel format) for the human body model (HBM), defining joint orientations and positions, and joint ranges of motion (ROMs).
- **Load 3D keypoints**: Load 3D pose keypoints (in `.txt` format).
- **Inverse kinematics**: Perform frame-based or temporal IK optimisation to align the kinematic chain with the loaded 3D pose keypoints.
- **Motion smoothing**: Apply the Savitzky-Golay filter to smooth IK results across frames.
- **Visualisation**: Visualise both the kinematic chain and the pose data, with options to display animations using a slider.
- **Save results**: Save the transformation data (local/global orientations, positions, and angular velocities) to Excel files for further analysis.

Video demo available here: [VideoDemo_x175.mp4](Demo/VideoDemo_x175.mp4).

### Outputs:
- **Transformation data**: Saves local and global joint orientations and positions in `.xlsx` files.
- **Angular velocities**: Saves angular velocities for each joint in `.xlsx` format.

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
2. Add user options for smoothing (user-defined, and informed by function value).
3. Fix threading issue.
4. Optional input of a custom pose.
5. Add linear positions and velocities to output.
6. Add live visualisation of the loss terms, and a final visualisation of convergence (plotting of guesses at various iterations for each frame accross the motion sequence)
