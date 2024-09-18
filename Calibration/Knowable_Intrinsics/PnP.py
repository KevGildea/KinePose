import tkinter as tk 
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk
import os

class PnPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PnP Tool")
        self.zoom_factor = 1.0  # Fixed zoom level at 1x (having issues with zooming)
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0

        # Create the main frame to hold both sides
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left side for image
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right side for 3D plot
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create a dropdown menu (replacing buttons)
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Load World Points", command=self.load_world_points)
        file_menu.add_command(label="Load .tacal (Intrinsics)", command=self.load_tacal_file)
        file_menu.add_command(label="Undo Last Annotation", command=self.undo_annotation)
        file_menu.add_command(label="Save Calibration", command=self.save_calibration)

        # Canvas for displaying the image on the left side
        self.canvas = tk.Canvas(self.left_frame, width=500, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize variables
        self.image = None
        self.world_points = None
        self.image_points = []
        self.K = None  # Intrinsic matrix
        self.extrinsic_matrix = None
        self.tacal_file = None
        self.image_name = None
        self.current_point_index = 0
        self.tacal_file_content = None
        self.photo_image = None
        self.original_image = None

        # Set up the 3D plot on the right side
        self.figure = plt.figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind right-click (Button-3) events for panning
        self.canvas.bind("<ButtonPress-3>", self.start_pan)  # Right-click to start panning
        self.canvas.bind("<B3-Motion>", self.pan_image)  # Drag with right-click to pan
        # Bind left-click (Button-1) for annotation
        self.canvas.bind("<Button-1>", self.annotate_image)  # Left-click to annotate

    def load_image(self):
        # Load an image
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg")])
        if image_path:
            self.image = cv2.imread(image_path)
            if self.image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return
            self.image_name = os.path.basename(image_path).split('.')[0]
            self.original_image = self.image.copy()  # Keep a copy of the original image for panning
            self.image_points.clear()  # Reset previous annotations
            self.current_point_index = 0  # Start with the first world point
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            messagebox.showinfo("Loaded", "Image loaded successfully.")
            self.update_image_canvas()

    def load_world_points(self):
        # Load world coordinates from a space-separated .txt file
        world_points_path = filedialog.askopenfilename(title="Select World Points", filetypes=[("Text Files", "*.txt")])
        if world_points_path:
            try:
                self.world_points = np.loadtxt(world_points_path, delimiter=" ")
                if self.world_points.shape[1] != 3:
                    raise ValueError("Invalid world coordinates format. Must have 3 columns (X, Y, Z).")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load world points: {e}")
                return
            messagebox.showinfo("Loaded", "World points loaded successfully.")
            self.plot_world_points_3d()

    def load_tacal_file(self):
        # Load the .tacal file (with intrinsic parameters) and parse it
        tacal_file_path = filedialog.askopenfilename(title="Select .tacal File", filetypes=[("TACAL Files", "*.tacal")])
        if tacal_file_path:
            with open(tacal_file_path, 'r') as f:
                self.tacal_file_content = f.readlines()
                self.K = self.parse_intrinsics(self.tacal_file_content)
                self.tacal_file = tacal_file_path
            if self.K is not None:
                messagebox.showinfo("Loaded", ".tacal file loaded successfully.")
            else:
                messagebox.showerror("Error", "Failed to load intrinsic parameters from .tacal file.")

    def parse_intrinsics(self, content):
        # Parse the intrinsic matrix from the .tacal file, handle 'None' entries
        intrinsics = {}
        for line in content:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if value == 'None':
                    continue
                intrinsics[key] = float(value)
        
        try:
            fx = fy = intrinsics["f"]
            cx = intrinsics["Cx"]
            cy = intrinsics["Cy"]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        except KeyError:
            messagebox.showerror("Error", "Failed to extract intrinsic parameters from .tacal file.")
            return None

    def update_image_canvas(self):
        # Resize and display the current image on the canvas
        if self.image is None:
            return
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        resized_image = cv2.resize(rgb_image, (int(self.original_image.shape[1] * self.zoom_factor),
                                               int(self.original_image.shape[0] * self.zoom_factor)))  # Resize for zoom

        # Ensure pan offsets stay within image boundaries
        self.pan_offset_x = max(0, min(self.pan_offset_x, resized_image.shape[1] - 500))
        self.pan_offset_y = max(0, min(self.pan_offset_y, resized_image.shape[0] - 500))

        # Apply pan offsets
        x_start = int(self.pan_offset_x)
        y_start = int(self.pan_offset_y)
        x_end = min(resized_image.shape[1], x_start + 500)
        y_end = min(resized_image.shape[0], y_start + 500)
        cropped_image = resized_image[y_start:y_end, x_start:x_end]

        # Convert the image to PIL format
        pil_image = Image.fromarray(cropped_image)
        
        # Convert the PIL image to an ImageTk.PhotoImage object for tkinter
        self.photo_image = ImageTk.PhotoImage(pil_image)

        # Update the canvas with the image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.draw_annotations()  # Draw annotations after updating the image

    def plot_world_points_3d(self):
        # Plot the loaded world points in 3D with equal axes on the right
        self.ax.clear()  # Clear any previous plot
        self.ax.scatter(self.world_points[:, 0], self.world_points[:, 1], self.world_points[:, 2], color='blue', s=50)

        # Label each point with its index
        for i, point in enumerate(self.world_points):
            self.ax.text(point[0], point[1], point[2], str(i), color='red')

        # Set axis labels and equal scaling
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('World Points')

        # Set equal axes scaling
        max_range = np.array([self.world_points[:, 0].max() - self.world_points[:, 0].min(),
                              self.world_points[:, 1].max() - self.world_points[:, 1].min(),
                              self.world_points[:, 2].max() - self.world_points[:, 2].min()]).max() / 2.0

        mid_x = (self.world_points[:, 0].max() + self.world_points[:, 0].min()) * 0.5
        mid_y = (self.world_points[:, 1].max() + self.world_points[:, 1].min()) * 0.5
        mid_z = (self.world_points[:, 2].max() + self.world_points[:, 2].min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.canvas_plot.draw()

    def annotate_image(self, event):
        # Store clicked image points and move to the next point
        if self.current_point_index < len(self.world_points):
            # Adjust the click coordinates based on fixed zoom (x3) and pan
            x = int((event.x / self.zoom_factor) + self.pan_offset_x)
            y = int((event.y / self.zoom_factor) + self.pan_offset_y)

            self.image_points.append([x, y])
            self.update_image_canvas()  # Update the canvas with the annotation
            self.current_point_index += 1

            if self.current_point_index == len(self.world_points):
                messagebox.showinfo("Completed", "All points have been annotated.")
        else:
            messagebox.showinfo("Completed", "All points have already been annotated.")

    def undo_annotation(self):
        # Undo the last annotation
        if self.image_points:
            self.image_points.pop()
            self.current_point_index -= 1
            # Redraw the image without the last annotation
            self.update_image_canvas()
        else:
            messagebox.showinfo("Undo", "No annotations to undo.")

    def save_calibration(self):
        # Generate extrinsic matrix and save to .tacal
        if self.K is None:
            messagebox.showerror("Error", "Intrinsic matrix not loaded. Please load the .tacal file.")
            return

        if len(self.image_points) != len(self.world_points):
            messagebox.showerror("Error", "Not all image points have been annotated.")
            return

        # Solve for extrinsics
        world_points = np.array(self.world_points, dtype=np.float32)
        image_points = np.array(self.image_points, dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(world_points, image_points, self.K, distCoeffs=None)
        R, _ = cv2.Rodrigues(rvec)
        
        # Check for valid rotation matrix
        if not np.isclose(np.linalg.det(R), 1.0):
            messagebox.showerror("Error", "Invalid rotation matrix")
            return
        
        # Stack R and tvec into extrinsics matrix
        extrinsics = np.hstack((R, tvec))

        # Compute the reprojection error
        projected_image_points, _ = cv2.projectPoints(world_points, rvec, tvec, self.K, distCoeffs=None)
        projected_image_points = projected_image_points.reshape(-1, 2)
        errors = np.linalg.norm(image_points - projected_image_points, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # Save reprojection error to a log file
        self.save_reprojection_error_log(mean_error, max_error, errors)

        # Update the .tacal file with the correct order of extrinsics
        self.update_tacal_file(R, tvec)

    def save_reprojection_error_log(self, mean_error, max_error, errors):
        # Create a log file with reprojection errors
        base_filename = os.path.splitext(self.tacal_file)[0]  # Get the filename without extension
        log_filename = f"{base_filename}_reprojection_error.log"  # Create a log filename

        # Write the errors to the log file
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Mean Reprojection Error: {mean_error:.6f}\n")
            log_file.write(f"Max Reprojection Error: {max_error:.6f}\n")
            log_file.write("Individual Reprojection Errors:\n")
            for i, error in enumerate(errors):
                log_file.write(f"Point {i}: {error:.6f}\n")

        messagebox.showinfo("Saved", f"Reprojection error log saved to '{log_filename}'")

    def update_tacal_file(self, R, tvec):
        # Modify the existing .tacal file by replacing the 'None' placeholders
        new_content = []
        
        # Extract rotation and translation components separately
        rotation_flat = R.flatten()  # r1 to r9
        translation_flat = tvec.flatten()  # Tx, Ty, Tz

        # Update tacal file with correct mapping
        for line in self.tacal_file_content:
            key = line.split(":")[0].strip()
            
            if key == "Tx":
                new_content.append(f"{key}:      {translation_flat[0]:.6f}\n")
            elif key == "Ty":
                new_content.append(f"{key}:      {translation_flat[1]:.6f}\n")
            elif key == "Tz":
                new_content.append(f"{key}:      {translation_flat[2]:.6f}\n")
            elif key == "r1":
                new_content.append(f"{key}:      {rotation_flat[0]:.6f}\n")
            elif key == "r2":
                new_content.append(f"{key}:      {rotation_flat[1]:.6f}\n")
            elif key == "r3":
                new_content.append(f"{key}:      {rotation_flat[2]:.6f}\n")
            elif key == "r4":
                new_content.append(f"{key}:      {rotation_flat[3]:.6f}\n")
            elif key == "r5":
                new_content.append(f"{key}:      {rotation_flat[4]:.6f}\n")
            elif key == "r6":
                new_content.append(f"{key}:      {rotation_flat[5]:.6f}\n")
            elif key == "r7":
                new_content.append(f"{key}:      {rotation_flat[6]:.6f}\n")
            elif key == "r8":
                new_content.append(f"{key}:      {rotation_flat[7]:.6f}\n")
            elif key == "r9":
                new_content.append(f"{key}:      {rotation_flat[8]:.6f}\n")
            else:
                new_content.append(line)

        # Create a new filename by adding '_extrinsics' to the original name
        base_filename = os.path.splitext(self.tacal_file)[0]  # Get the filename without extension
        new_filename = f"{base_filename}_extrinsics.tacal"    # Append '_extrinsics' to the filename

        # Save the modified content to a new .tacal file
        with open(new_filename, 'w') as f:
            f.writelines(new_content)

        messagebox.showinfo("Saved", f"Extrinsics saved to '{new_filename}' successfully.")



    def start_pan(self, event):
        # Initialize panning by recording the start position
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_image(self, event):
        # Pan the image when the right mouse button is held and dragged
        dx = self.pan_start_x - event.x
        dy = self.pan_start_y - event.y
        self.pan_offset_x += dx
        self.pan_offset_y += dy

        # Update the start positions for the next movement
        self.pan_start_x = event.x
        self.pan_start_y = event.y

        self.update_image_canvas()

    def draw_annotations(self):
        # Draw all the annotations on the canvas
        for i, (x, y) in enumerate(self.image_points):
            # Calculate the position of the annotation based on zoom and pan
            x_on_canvas = (x - self.pan_offset_x) * self.zoom_factor
            y_on_canvas = (y - self.pan_offset_y) * self.zoom_factor

            # Only draw the annotation if it's within the current canvas view
            if 0 <= x_on_canvas <= 500 and 0 <= y_on_canvas <= 500:
                # Draw the point
                self.canvas.create_oval(x_on_canvas - 3, y_on_canvas - 3, 
                                        x_on_canvas + 3, y_on_canvas + 3, 
                                        outline="green", fill="green")

                # Draw the index (as text next to the point)
                self.canvas.create_text(x_on_canvas + 10, y_on_canvas, 
                                        text=str(i), fill="red")


# Main application loop
if __name__ == "__main__":
    root = tk.Tk()
    app = PnPApp(root)
    root.mainloop()
