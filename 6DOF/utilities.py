# Import necessary modules
import numpy as np
import tkinter as tk
from scipy.signal import savgol_filter
import pandas as pd

class utilities:

    def root2origin(prediction_w_global):
        prediction_w_global_aligned=[]
        for i in range(len(prediction_w_global)):
            prediction_w_global_aligned_frame=[]
            for j in range(len(prediction_w_global[i])):
                prediction_w_global_aligned_frame.append(list(prediction_w_global[i][j]-prediction_w_global[0][0]))
            prediction_w_global_aligned.append(prediction_w_global_aligned_frame)
        return prediction_w_global_aligned

    def smooth_weights(weights, max_window_length=5, polyorder=2):
        num_frames = weights.shape[0]
        
        # adaptive window length based on the number of frames
        window_length = max_window_length if num_frames > max_window_length else num_frames - (num_frames % 2) - 1
        window_length = max(window_length, 3)  # ensure min window length of 3
        poly_order_adjusted = min(polyorder, window_length - 1)
        
        smoothed_weights = np.copy(weights)
        
        for i in range(weights.shape[1]):
            if num_frames > poly_order_adjusted:
                # apply filter only if there are enough frames
                smoothed_weights[:, i] = savgol_filter(weights[:, i], window_length, poly_order_adjusted)
        
        return smoothed_weights



    def save_transformation_data_to_excel(rotation_matrices, position_vectors, filename):
        frames, joints, _, _ = rotation_matrices.shape
        data_rows = []

        # column labels
        columns = []
        for joint in range(joints):
            for i in range(3):
                for j in range(3):
                    columns.append(f"Joint{joint}_RotMat_{i+1}{j+1}")
            for k in range(3):
                columns.append(f"Joint{joint}_PosVec_{k+1}")

        # flatten
        for frame in range(frames):
            row = []
            for joint in range(joints):
                row.extend(rotation_matrices[frame, joint].flatten())
                row.extend(position_vectors[frame, joint].flatten())
            data_rows.append(row)
        
        # save
        df = pd.DataFrame(data_rows, columns=columns)
        df.to_excel(filename, index_label="Frame")


    def save_angular_velocities_to_excel(angular_velocities, filename):

        frames, joints, _ = angular_velocities.shape
        data_rows = []

        # column labels for angular velocity components
        columns = []
        for joint in range(joints):
            columns.append(f"Joint{joint}_AngularVelocity_X")
            columns.append(f"Joint{joint}_AngularVelocity_Y")
            columns.append(f"Joint{joint}_AngularVelocity_Z")
        
        # flatten
        for frame in range(frames):
            row = []
            for joint in range(joints):
                row.extend(angular_velocities[frame, joint])
            data_rows.append(row)
        # save
        df = pd.DataFrame(data_rows, columns=columns)
        df.to_excel(filename, index_label="Frame")


    def calculate_angular_velocities(rotation_matrices, delta_t):
        """
        Calculates angular velocities using central difference from a series of rotation matrices.
        
        Parameters:
        rotation_matrices : numpy array
            A 4D array of shape (frames, joints, 3, 3) containing rotation matrices.
        delta_t : float
            Time step between frames.
        
        Returns:
        angular_velocities : numpy array
            A 3D array of shape (frames, joints, 3) containing angular velocity vectors, with None
            for the first and last frames.
        """
        frames, joints, _, _ = rotation_matrices.shape
        angular_velocities = np.full((frames, joints, 3), None)  # Initialize with None

        for joint in range(joints):
            for frame in range(1, frames - 1):  # Skip the first and last frames
                # Central difference: use the next and previous frames
                R_next = np.dot(np.linalg.inv(rotation_matrices[frame, joint]), rotation_matrices[frame + 1, joint])
                R_prev = np.dot(np.linalg.inv(rotation_matrices[frame - 1, joint]), rotation_matrices[frame, joint])
                
                # Compute angular velocity using the central difference method
                angular_velocity = (utilities.log_so3(R_next) - utilities.log_so3(R_prev)) / (2 * delta_t)
                
                angular_velocities[frame, joint] = angular_velocity

        return angular_velocities

    def log_so3(R):
        """
        Compute the matrix logarithm of a rotation matrix in SO(3).
        
        Parameters:
        R : numpy array
            A 3x3 rotation matrix.
        
        Returns:
        omega : numpy array
            A 3-element array representing the angular velocity vector.
        """
        assert np.allclose(np.dot(R.T, R), np.eye(3), atol=1e-6), "R is not a valid rotation matrix"
        assert np.isclose(np.linalg.det(R), 1.0), "R does not have a determinant of 1"
        
        # Compute the matrix logarithm via Rodrigues' formula
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        if np.isclose(theta, 0):  # No rotation
            return np.zeros(3)
        else:
            return theta / (2 * np.sin(theta)) * np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ])

# custom class to redirect stdout
class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # auto-scroll to the end

    def flush(self):
        pass
