import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import widgets
from matplotlib.animation import FuncAnimation
import glob
import os
import re
from scipy.signal import savgol_filter

def natural_sort(file_list):
    """ Sorts filenames containing numbers in natural order. """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    return sorted(file_list, key=natural_keys)

# Mapping from COCO to H36M
COCO_TO_H36M_MAPPING = {
    #0: 9,   # Nose -> Head
    5: 10,  # Left Shoulder -> Left Shoulder
    6: 13,  # Right Shoulder -> Right Shoulder
    7: 11,  # Left Elbow -> Left Elbow
    8: 14,  # Right Elbow -> Right Elbow
    9: 12,  # Left Wrist -> Left Wrist
    10: 15, # Right Wrist -> Right Wrist
    11: 4,  # Left Hip -> Left Hip
    12: 1,  # Right Hip -> Right Hip
    13: 5,  # Left Knee -> Left Knee
    14: 2,  # Right Knee -> Right Knee
    15: 6,  # Left Ankle -> Left Ankle
    16: 3   # Right Ankle -> Right Ankle
}

def map_coco_to_h36m(coco_keypoints):
    """
    Map COCO keypoints to H3.6M keypoints based on the COCO_TO_H36M_MAPPING dictionary,
    with special handling for head, neck, and mid-hip.
    """
    h36m_keypoints = np.zeros((16, 3))  # H3.6M format has 17 joints
    
    # Set the head as the average of the nose, eyes, and ears
    nose = coco_keypoints[0]
    left_eye = coco_keypoints[1]
    right_eye = coco_keypoints[2]
    left_ear = coco_keypoints[3]
    right_ear = coco_keypoints[4]
    
    # Set head keypoint
    h36m_keypoints[9] = np.mean([nose, left_eye, right_eye, left_ear, right_ear], axis=0)  # Head (nose, left_eye, right_eye, )
    
    # Map all other joints based on COCO_TO_H36M_MAPPING
    for coco_idx, h36m_idx in COCO_TO_H36M_MAPPING.items():
        h36m_keypoints[h36m_idx] = coco_keypoints[coco_idx]
    
    # Set the Hip (Center) as the average of Left and Right Hips
    h36m_keypoints[0] = (h36m_keypoints[1] + h36m_keypoints[4]) / 2  # Mid-Hip
    
    # Set the Neck as the midpoint of Left and Right Shoulders
    h36m_keypoints[8] = (h36m_keypoints[10] + h36m_keypoints[13]) / 2  # Neck

    # Set the Spine as the midpoint of the Neck and Mid-Hip
    h36m_keypoints[7] = (h36m_keypoints[0] + h36m_keypoints[8]) / 2  # Spine

    # Remove keypoint 16 and 7 (problematic static point)
    #h36m_keypoints = np.delete(h36m_keypoints, (16), axis=0)
    #h36m_keypoints = np.delete(h36m_keypoints, 7, axis=0)
    
    return h36m_keypoints

def read_calibration(file_path):
    """Reads a camera calibration file and extracts parameters."""
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.split(':')
                params[key.strip()] = float(value.strip())
    return params

def construct_matrices(params):
    """Constructs the intrinsic and extrinsic matrices from the parameters."""
    # Intrinsic matrix K
    f = params['f']
    Cx = params['Cx']
    Cy = params['Cy']
    K = np.array([[f, 0, Cx], [0, f, Cy], [0, 0, 1]])

    # Extrinsic parameters (rotation matrix R and translation vector T)
    R = np.array([
        [params['r1'], params['r2'], params['r3']],
        [params['r4'], params['r5'], params['r6']],
        [params['r7'], params['r8'], params['r9']]
    ])
    T = np.array([params['Tx'], params['Ty'], params['Tz']])

    # Compute the inverse transformation for visualization
    R_inv = R.T
    T_inv = -R_inv @ T

    # Combine into a projection matrix
    P = K @ np.hstack((R, T.reshape(-1, 1)))
    
    return K, R_inv, T_inv, P


def triangulate_weighted_midpoint(P1, P2, pts1, pts2, weights1=None, weights2=None):
    """
    Triangulate points using a Weighted Midpoint Method, incorporating confidence scores as weights (Not yet implemented here - set to None).

    :param P1: 3x4 projection matrix for the first camera.
    :param P2: 3x4 projection matrix for the second camera.
    :param pts1: 2xN array of points in the first image.
    :param pts2: 2xN array of points in the second image.
    :param weights1: 1xN array of weights for the points in the first image.
    :param weights2: 1xN array of weights for the points in the second image.
    :return: 3xN array of triangulated 3D points.
    """
    num_points = pts1.shape[1]
    output_points = np.zeros((3, num_points))

    if weights1 is None:
        weights1 = np.ones(num_points)
    if weights2 is None:
        weights2 = np.ones(num_points)

    for i in range(num_points):
        # Extract point coordinates
        x1, y1 = pts1[:, i]
        x2, y2 = pts2[:, i]

        # Extract weights
        w1 = weights1[i]
        w2 = weights2[i]

        # Build matrices A and b for the least squares solution
        A = np.vstack([
            w1 * (x1 * P1[2, :] - P1[0, :]),
            w1 * (y1 * P1[2, :] - P1[1, :]),
            w2 * (x2 * P2[2, :] - P2[0, :]),
            w2 * (y2 * P2[2, :] - P2[1, :])
        ])
        b = np.zeros(4)  # No need for b as A should already align the equations

        # Solve the least squares problem
        X, residuals, rank, s = np.linalg.lstsq(A[:, :3], -A[:, 3], rcond=None)

        # Store the solution
        output_points[:, i] = X

    return output_points


def read_keypoints(file_path):
    keypoints = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                keypoint, coordinates = parts
                try:
                    x, y = coordinates.strip().split(',')
                    # Convert x and y to float instead of int
                    keypoints[keypoint.strip()] = (float(x), float(y))
                except ValueError as e:
                    print(f"Error parsing line '{line.strip()}': {e}")
    return keypoints

def save_keypoints_to_file(file_path, keypoints_list):
    """
    Save the 3D keypoints to a .txt file in the required format, with an extra keypoint (0, 0, 0) added at index 9.

    :param file_path: The path to the file where the keypoints should be saved.
    :param keypoints_list: List of 3D keypoints for each frame.
    """
    with open(file_path, 'w') as f:
        for keypoints in keypoints_list:
            # Convert the keypoints array to a list so we can insert
            keypoints_with_extra = list(keypoints)

            # Insert the extra (0, 0, 0) point at index 9 (for compatibility with KinePose tool)
            keypoints_with_extra.insert(9, [0.0, 0.0, 0.0])

            # Write the keypoints to the file
            for point in keypoints_with_extra:
                x, y, z = point
                f.write(f"{x} {y} {z}\n")


def apply_savgol_filter(poses, window_length=30, polyorder=2):
    """
    Applies the Savitzky-Golay filter to smooth 3D keypoints over time.
    :param poses: List of 3D keypoints for each frame
    :param window_length: The window length of the filter
    :param polyorder: The polynomial order of the filter
    :return: Smoothed keypoints (as numpy arrays)
    """
    poses_np = np.array(poses)  # Convert the input to a numpy array
    
    # Apply Savitzky-Golay filter along each axis separately
    smoothed_poses = savgol_filter(poses_np, window_length, polyorder, axis=0)
    
    return smoothed_poses  # Return the smoothed numpy array

class Pose3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("wMP triangulation")

        # Create a menu bar with an "About" section
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        # Create a frame for file selection and options
        frame = tk.Frame(root)
        frame.pack(pady=10)

        # Buttons to load camera calibration and keypoint data
        self.load_calib_button = tk.Button(frame, text="Load Camera Calibration", command=self.load_calibration)
        self.load_calib_button.pack(side=tk.LEFT, padx=5)

        self.load_keypoints_button = tk.Button(frame, text="Load Keypoints Folders", command=self.load_keypoints_folders)
        self.load_keypoints_button.pack(side=tk.LEFT, padx=5)

        # Initialize paths and variables
        self.camera1_params = None
        self.camera2_params = None
        self.keypoints_folder1 = None
        self.keypoints_folder2 = None

    # about me :)
    def show_about(self):
        about_window = tk.Toplevel(root)
        about_window.title("About")

        info = (
            "Developed by Kevin Gildea, Ph.D.\n"
            "Faculty of Engineering, LTH\n"
            "Lund University\n"
            "Email: kevin.gildea@tft.lth.se"
        )

        text_widget = tk.Text(about_window, height=5, width=27, wrap="word", font=("Arial", 8))
        text_widget.insert("1.0", info)
        text_widget.config(state="disabled", bg=about_window.cget("bg"))
        text_widget.pack(pady=15)

        # text selectable and copyable but not editable
        text_widget.bind("<1>", lambda event: text_widget.focus_set())

    def load_calibration(self):
        # Load camera calibration files
        self.camera1_path = filedialog.askopenfilename(title="Select Camera 1 Calibration")
        self.camera2_path = filedialog.askopenfilename(title="Select Camera 2 Calibration")
        
        if self.camera1_path and self.camera2_path:
            self.camera1_params = read_calibration(self.camera1_path)
            self.camera2_params = read_calibration(self.camera2_path)
            messagebox.showinfo("Loaded", "Calibration files loaded successfully!")
        else:
            messagebox.showwarning("Error", "Please select both calibration files.")

    def load_keypoints_folders(self):
        # Load keypoints folders for both cameras
        self.keypoints_folder1 = filedialog.askdirectory(title="Select Folder for Camera 1 Keypoints")
        self.keypoints_folder2 = filedialog.askdirectory(title="Select Folder for Camera 2 Keypoints")

        if self.keypoints_folder1 and self.keypoints_folder2:
            messagebox.showinfo("Loaded", "Keypoints folders loaded successfully!")
            self.visualize_pose_3d_with_slider()
        else:
            messagebox.showwarning("Error", "Please select both keypoints folders.")

    def visualize_pose_3d_with_slider(self):
        # Visualize the 3D pose using sliders and animation
        if not self.camera1_params or not self.camera2_params or not self.keypoints_folder1 or not self.keypoints_folder2:
            messagebox.showerror("Error", "Missing calibration or keypoint folder data.")
            return

        # Construct camera matrices
        K1, R1_inv, T1_inv, P1 = construct_matrices(self.camera1_params)
        K2, R2_inv, T2_inv, P2 = construct_matrices(self.camera2_params)

        # Use glob to read all files from the folders in sequence
        keypoints_files1 = natural_sort(glob.glob(os.path.join(self.keypoints_folder1, "*.txt")))
        keypoints_files2 = natural_sort(glob.glob(os.path.join(self.keypoints_folder2, "*.txt")))

        if len(keypoints_files1) != len(keypoints_files2):
            messagebox.showerror("Error", "Mismatched number of keypoint files between cameras.")
            return

        # Store the 3D poses for each frame
        pos_b = []
        coco_pos_b = []  # To store raw COCO triangulated data for plotting

        for file1, file2 in zip(keypoints_files1, keypoints_files2):
            # Read the 2D keypoints for each frame
            points1 = np.array(list(read_keypoints(file1).values())).T
            points2 = np.array(list(read_keypoints(file2).values())).T

            # Triangulate 3D points for each frame using your triangulation method
            points3D = triangulate_weighted_midpoint(P1, P2, points1, points2)

            # Store the raw triangulated COCO points for visualization
            coco_pos_b.append(points3D.T)

            # Map the triangulated points from COCO to H3.6M format
            h36m_pose = map_coco_to_h36m(points3D.T)
            
            # Normalize to root (mid-hip)
            #normalized_h36m_pose = normalize_to_root(h36m_pose)
            
            pos_b.append(h36m_pose)
        
        pos_b = apply_savgol_filter(pos_b)

        dir_graph = {
            0: [1, 4, 7, 8],  # Mid-Hip -> Left Hip, Right Hip, Spine, Neck
            1: [2],           # Left Hip -> Left Knee
            2: [3],           # Right Hip -> Right Knee
            4: [5],           # Right Knee -> Right Ankle
            5: [6],           # Left Knee -> Left Ankle
            8: [9, 10, 13],   # Neck -> Head, Left Shoulder, Right Shoulder
            10: [11],         # Left Shoulder -> Left Elbow
            11: [12],         # Left Elbow -> Left Wrist
            13: [14],         # Right Shoulder -> Right Elbow
            14: [15]          # Right Elbow -> Right Wrist
        }

        # Pass the frames (pos_b) to the plot with sliders and animation
        self.plot_pose_local_aniplusslider(pos_b, coco_pos_b, dir_graph, 0, len(pos_b), title='wMP triangulation')
        
    def plot_pose_local_aniplusslider(self, pos_b, coco_pos_b, dir_graph, start, stop, title=''):
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title, fontsize=12)

        ax_slider_plot = fig.add_subplot(121, projection='3d')
        ax_slider_plot.set_axis_off()

        ax_animation_plot = fig.add_subplot(122, projection='3d')
        ax_animation_plot.set_axis_off()

        def plot_frame(ax, frame, show_frame_number=False):
            ax.clear()

            # Plot H36M skeleton points and add labels for each keypoint ID
            x1, y1, z1 = pos_b[frame][:, 0], pos_b[frame][:, 1], pos_b[frame][:, 2]
            ax.scatter(x1, y1, z1, c='blue', s=100, marker='o')

            # Label each keypoint with its index for H36M
            #for idx in range(len(x1)):
            #     ax.text(x1[idx], y1[idx], z1[idx], f'{idx}', color='blue', fontsize=16)

            # Plot the MSCOCO triangulated data in red (no skeleton) and label keypoint IDs
            coco_x, coco_y, coco_z = coco_pos_b[frame][:, 0], coco_pos_b[frame][:, 1], coco_pos_b[frame][:, 2]
            ax.scatter(coco_x, coco_y, coco_z, c='red', s=50, marker='o',alpha=0.3)  # No skeleton, just raw points

            # Label each keypoint with its index for COCO
            #for idx in range(len(coco_x)):
            #    ax.text(coco_x[idx], coco_y[idx], coco_z[idx], f'{idx}', color='red', fontsize=10)

            # Draw H36M skeleton lines
            parent_child_pairs = [(k, v) for k, children in dir_graph.items() for v in children]
            for j in range(len(parent_child_pairs)):
                xs = [x1[parent_child_pairs[j][0]], x1[parent_child_pairs[j][1]]]
                ys = [y1[parent_child_pairs[j][0]], y1[parent_child_pairs[j][1]]]
                zs = [z1[parent_child_pairs[j][0]], z1[parent_child_pairs[j][1]]]
                line = art3d.Line3D(xs, ys, zs, linewidth=6, c='gray', linestyle='--')
                ax.add_line(line)

            # Dynamically adjust the axis limits based on the current frame's points
            all_x = np.concatenate((x1, coco_x))
            all_y = np.concatenate((y1, coco_y))
            all_z = np.concatenate((z1, coco_z))

            buffer = 0.1  # Add some buffer to the limits to avoid tight zoom

            # Find the overall range of the data
            max_range = np.array([max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]).max() / 2.0

            # Find the midpoints of the data
            mid_x = (max(all_x) + min(all_x)) * 0.5
            mid_y = (max(all_y) + min(all_y)) * 0.5
            mid_z = (max(all_z) + min(all_z)) * 0.5

            # Set the limits equally for all axes
            ax.set_xlim3d(mid_x - max_range - buffer, mid_x + max_range + buffer)
            ax.set_ylim3d(mid_y - max_range - buffer, mid_y + max_range + buffer)
            ax.set_zlim3d(mid_z - max_range - buffer, mid_z + max_range + buffer)

            ax.set_xlabel('Global X')
            ax.set_ylabel('Global Y')
            ax.set_zlabel('Global Z')
            #ax.set_axis_off()

            if show_frame_number:
                ax.text2D(0.05, 0.95, f"Frame: {frame}", transform=ax.transAxes)

        def update_slider(val):
            frame = int(val)
            plot_frame(ax_slider_plot, frame)

        ax_frame_slider = plt.axes([0.15, 0.02, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        frame_slider = widgets.Slider(ax_frame_slider, 'Frame', start, stop - 1, valinit=start, valfmt='%0.0f')
        frame_slider.on_changed(update_slider)

        ani = [None]  # Encapsulate the animation in a list to allow modification

        def create_animation(interval):
            if ani[0]:
                ani[0].event_source.stop()  # Stop the previous animation
            ani[0] = FuncAnimation(fig, lambda frame: plot_frame(ax_animation_plot, frame, show_frame_number=True),
                                frames=range(start, stop), interval=interval, repeat=True)

        def update_interval(val):
            interval = int(val)
            create_animation(interval)

        ax_interval_slider = plt.axes([0.6, 0.02, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        interval_slider = widgets.Slider(ax_interval_slider, 'Interval (ms)', 10, 1000, valinit=500, valfmt='%0.0f ms')
        interval_slider.on_changed(update_interval)

        update_slider(start)
        create_animation(500)  # Initialize the animation with the default interval

        # Save the H36M 3D keypoints to a file
        save_keypoints_to_file("3D_keypoints.txt", pos_b)

        plt.show()


# Main application loop
if __name__ == "__main__":
    root = tk.Tk()
    app = Pose3DApp(root)
    root.mainloop()
