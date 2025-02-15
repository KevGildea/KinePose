## [3DMonocular.exe](Dist/3DMonocular.exe)

This tool reconstructs 3D human poses from 2D pose sequences using deep learning models adapted from the [MotionBERT](https://github.com/Walter0807/MotionBERT) framework. The input is be 2D pose keypoints extracted from images or video using the KinePose [2D pose tools](https://github.com/KevGildea/KinePose/tree/main/2D), and the output is a 3D pose representation. Run on a Windows machine as administrator, and include accompanying `.bin`, and `.yaml` files in the same directory.

### Key features:

- **Load 2D poses and video/image**: Select a folder with 2D poses and a video or image file for processing.
- **Monocular 3D pose estimation**: Reconstructs 3D poses from a single-camera (monocular) setup.
- **Deep learning model**: Utilises a pre-trained MotionBERT model for pose estimation.
- **Video/image output**: Outputs the 3D pose visualisation as a video (`.mp4`) or an image (`.png`) depending on the input type.
- **Keypoint output**: Saves the 3D keypoints for each frame to a `.txt` file in a format compatible with further processing in KinePose.

### Outputs:
- **3D pose data**: Saves the 3D pose keypoints for each frame in a single `.txt` file.
- **Visualisation**: Renders and saves the 3D pose estimate as a video or image.


### GUI
<p align="center">
  <img src="https://github.com/KevGildea/KinePose/blob/main/images/3DMonocular.PNG" alt="3DMonocular.exe" width="400">
  <br>
  <i>The GUI of the 3DMonocular.exe application.</i>
</p>

## Contributing
Feel free to submit pull requests or open issues for improvements or bug fixes!

### To Do:
- [ ] Adapt to input commonly used 2D pose formats.
- [ ] Ensure 'results' outputs are always saved in a consistent manner accross all tools.
- [ ] Add visualisation of 3D pose.
- [ ] Fix threading issue.
