## [3DMonocular.exe](Dist/3DMonocular.exe)

This tool reconstructs 3D human poses from 2D pose sequences using deep learning models adapted from the [MotionBERT](https://github.com/Walter0807/MotionBERT) framework. The input can be 2D pose keypoints extracted from images or video, and the output is a 3D pose representation. Run on a Windows machine as administrator, and include accompanying `.bin`, and `.yaml` files in the same directory.

### Key features:

- **Load 2D Poses and Video/Image**: Select a folder with 2D poses and a video or image file for processing.
- **Monocular 3D Pose Estimation**: Reconstructs 3D poses from a single-camera (monocular) setup.
- **Deep Learning Model**: Utilises a pre-trained MotionBERT model for pose estimation.
- **Video/Image Output**: Outputs the 3D pose visualisation as a video (`.mp4`) or an image (`.png`) depending on the input type.
- **Keypoint Output**: Saves the 3D keypoints for each frame to a `.txt` file in a format compatible with further processing in KinePose.

### Outputs:
- **3D Pose Data**: Saves the 3D pose keypoints for each frame in a `.txt` file.
- **Visualisation**: Renders and saves the 3D pose reconstruction as a video or image.


### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/3DMonocular.PNG" alt="3DMonocular.exe" width="400">
  <br>
  <i>The GUI of the 3DMonocular.exe application.</i>
</p>


## To Do:
1. 
