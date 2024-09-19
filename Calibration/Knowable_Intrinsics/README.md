## Intrinsics.exe

This tool performs camera calibration using images of a checkerboard pattern. The calibration process computes the intrinsic camera parameters, including focal lengths and distortion coefficients. Key features:

- **Select Image Directory**: Choose a folder containing calibration images in `.png` format.
- **Checkerboard Specifications**: The tool uses a 10x7 checkerboard with 25mm square sizes. A file, `Checkerboard-A4-25mm-10x7.svg`, is included in the folder for you to download and print.
- **Automatic Checkerboard Detection**: Automatically detects checkerboard corners in the images.
- **Sub-pixel Refinement**: Improves the accuracy of the detected corners for better calibration results.
- **Intrinsic Parameter Calculation**: Computes the camera's intrinsic matrix and distortion coefficients based on the checkerboard data.
- **Save Calibration Data**: Saves the calibration results (intrinsic parameters) to a `.tacal` file in the selected folder, formatted for easy reuse.

### Outputs:
- **Intrinsic Parameters**:
  - Focal lengths (fx, fy)
  - Principal points (Cx, Cy)
  - Radial distortion coefficient (k)
- **File Output**: The intrinsic matrix is saved in a text format (`.tacal`), including pixel scaling and focal length data.







## To Do:
1. 
