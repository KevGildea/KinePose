## [Intrinsics.exe](Dist/Intrinsics.exe)

This tool performs camera calibration using images of a checkerboard pattern. The calibration process computes the intrinsic camera parameters, including focal lengths and distortion coefficients. Run on a Windows machine as administrator.

### Key features:

- **Select image directory**: Choose a folder containing calibration images in `.png` format.
- **Checkerboard specifications**: The tool uses a 10x7 checkerboard with 25mm square sizes. A file, `Checkerboard-A4-25mm-10x7.svg`, is included in the folder for you to download and print.
- **Automatic checkerboard detection**: Automatically detects checkerboard corners in the images.
- **Sub-pixel refinement**: Improves the accuracy of the detected corners for better calibration results.
- **Intrinsic parameter calculation**: Computes the camera's intrinsic matrix and distortion coefficients based on the checkerboard data.
- **Save calibration data**: Saves the calibration results (intrinsic parameters) to a `.tacal` file in the selected folder, formatted for later use.

### Outputs:
- **Intrinsic parameters**:
  - Focal lengths (fx, fy)
  - Principal points (Cx, Cy)
  - Radial distortion coefficient (k)
- **File Output**: The intrinsic matrix is saved in a text format (`.tacal`).

### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/Intrinsics.PNG" alt="Intrinsics.exe" width="250">
  <br>
  <i>The GUI of the Intrinsics.exe application.</i>
</p>




## [PnP.exe](Dist/PnP.exe)

This tool facilitates camera pose estimation using a set of 3D world points and corresponding 2D image points. The process calculates the extrinsic parameters of the camera (rotation and translation) using the **Perspective-n-Point** (PnP) algorithm. Run on a Windows machine as administrator.

### Key features:

- **Load image**: Select and display an image to annotate.
- **Load world points**: Load a set of 3D world coordinates from a `.txt` file.
- **Load .tacal (intrinsics)**: Load the camera's intrinsic parameters from a `.tacal` file.
- **Annotate image**: Manually annotate corresponding 2D image points by clicking on the loaded image.
- **Pan functionality**: Navigate the image while annotating.
- **Save calibration**: Save the computed extrinsic parameters (rotation, translation) to a new `.tacal` file with an added `_extrinsics` in the name, and log reprojection errors.

### Outputs:
- **Extrinsic parameters**:
  - Rotation matrix (R)
  - Translation vector (T)
- **File output**: The extrinsic matrix is saved in the `.tacal` format.
- **Reprojection error log**: Saves a log file containing the reprojection errors for each point, as well as the mean and maximum error.


### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/PnP.PNG" alt="PnP.exe" width="400">
  <br>
  <i>The GUI of the PnP.exe application.</i>
</p>





## Contributing
Feel free to submit pull requests or open issues for improvements or bug fixes!

### To Do:
- [ ] 
