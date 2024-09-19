## [Intrinsics.exe](Dist/Intrinsics.exe)

This tool performs camera calibration using images of a checkerboard pattern. The calibration process computes the intrinsic camera parameters, including focal lengths and distortion coefficients. Run on a Windows machine as administrator. 

### Key features:

- **Select Image Directory**: Choose a folder containing calibration images in `.png` format.
- **Checkerboard Specifications**: The tool uses a 10x7 checkerboard with 25mm square sizes. A file, `Checkerboard-A4-25mm-10x7.svg`, is included in the folder for you to download and print.
- **Automatic Checkerboard Detection**: Automatically detects checkerboard corners in the images.
- **Sub-pixel Refinement**: Improves the accuracy of the detected corners for better calibration results.
- **Intrinsic Parameter Calculation**: Computes the camera's intrinsic matrix and distortion coefficients based on the checkerboard data.
- **Save Calibration Data**: Saves the calibration results (intrinsic parameters) to a `.tacal` file in the selected folder, formatted for later use.

### Outputs:
- **Intrinsic Parameters**:
  - Focal lengths (fx, fy)
  - Principal points (Cx, Cy)
  - Radial distortion coefficient (k)
- **File Output**: The intrinsic matrix is saved in a text format (`.tacal`).


## [PnP.exe](Dist/PnP.exe)

This tool facilitates camera pose estimation using a set of 3D world points and corresponding 2D image points. The process calculates the extrinsic parameters of the camera (rotation and translation) using the **Perspective-n-Point** (PnP) algorithm. Run on a Windows machine as administrator.

### Key features:

- **Load Image**: Select and display an image to annotate.
- **Load World Points**: Load a set of 3D world coordinates from a `.txt` file.
- **Load .tacal (Intrinsics)**: Load the camera's intrinsic parameters from a `.tacal` file.
- **Annotate Image**: Manually annotate corresponding 2D image points by clicking on the loaded image.
- **Pan Functionality**: Navigate the image for precise annotations.
- **Save Calibration**: Save the computed extrinsic parameters (rotation, translation) to a new `.tacal` file with an added '_extrinsics' in the name, and log reprojection errors.

### Outputs:
- **Extrinsic Parameters**:
  - Rotation matrix (R)
  - Translation vector (T)
- **File Output**: The extrinsic matrix is saved in the `.tacal` format.
- **Reprojection Error Log**: Saves a log file containing the reprojection errors for each point, as well as the mean and maximum error.








## To Do:
1. 
