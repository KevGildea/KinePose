import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os

# Define the dimensions of the checkerboard
CHECKERBOARD = (10, 7)

# Stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the actual dimensions of the checkerboard (10x7 vertices, square size 25mm)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 25  # Multiply by 25mm


class CameraCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Calibration")

        # Create the UI
        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=10)

        self.select_dir_button = tk.Button(self.frame, text="Select Image Directory", command=self.select_directory)
        self.select_dir_button.pack(side=tk.LEFT, padx=10)

        self.calibrate_button = tk.Button(self.frame, text="Calibrate Camera", command=self.calibrate_camera)
        self.calibrate_button.pack(side=tk.LEFT, padx=10)

        self.intrinsic_matrix = None
        self.dist_coefficients = None
        self.directory = None

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        if self.directory:
            messagebox.showinfo("Directory Selected", f"Directory: {self.directory}")

    def calibrate_camera(self):
        if not self.directory:
            messagebox.showerror("Error", "Please select a directory with images")
            return

        # Arrays to store object points and image points from all images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Iterate over all images in the selected directory
        image_files = [f for f in os.listdir(self.directory) if f.endswith('.png')]
        if not image_files:
            messagebox.showerror("Error", "No images found in the directory")
            return

        for fname in image_files:
            img_path = os.path.join(self.directory, fname)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
                cv2.imshow('Detected Corners', img)

                # Wait for user to press a key to close the window
                cv2.waitKey(10)  # Adjust time to see the image (in milliseconds)

        cv2.destroyAllWindows()

        if len(objpoints) == 0:
            messagebox.showerror("Error", "No valid checkerboard images found")
            return

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Save the intrinsic matrix in the specified format using the folder name
        folder_name = os.path.basename(self.directory.rstrip('/\\'))
        output_file = os.path.join(self.directory, folder_name + "_intrinsics" ".tacal")
        self.save_intrinsics(output_file, mtx, dist)

        messagebox.showinfo("Calibration Complete", f"Intrinsic matrix saved as {output_file}")

    def save_intrinsics(self, output_file, mtx, dist):
        # Prepare data in the required format
        Cx, Cy = mtx[0, 2], mtx[1, 2]  # Principal points
        fx, fy = mtx[0, 0], mtx[1, 1]  # Focal lengths
        f = (fx + fy) / 2  # Average focal length
        dx, dy = 1, 1
        Sx = 1 / fx  # Pixel scaling in x direction (assuming dx = 1)

        # Write to file in the required format
        with open(output_file, 'w') as f_out:
            f_out.write(f"dx:      {dx}\n")
            f_out.write(f"dy:      {dy}\n")
            f_out.write(f"Cx:      {Cx}\n")
            f_out.write(f"Cy:      {Cy}\n")
            f_out.write(f"Sx:      {Sx}\n")
            f_out.write(f"f:      {f}\n")
            f_out.write(f"k:      {dist[0][0]}\n")  # Radial distortion coefficient
            f_out.write(f"Tx:      None\n")
            f_out.write(f"Ty:      None\n")
            f_out.write(f"Tz:      None\n")
            f_out.write(f"r1:      None\n")
            f_out.write(f"r2:      None\n")
            f_out.write(f"r3:      None\n")
            f_out.write(f"r4:      None\n")
            f_out.write(f"r5:      None\n")
            f_out.write(f"r6:      None\n")
            f_out.write(f"r7:      None\n")
            f_out.write(f"r8:      None\n")
            f_out.write(f"r9:      None\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCalibrationApp(root)
    root.mainloop()
