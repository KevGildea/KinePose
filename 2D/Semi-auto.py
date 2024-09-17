# Import necessary modules
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO
import os
from icon import icon_data
import base64
import os
import tempfile

# Convert the byte array back to bytes
icon_bytes = base64.b64decode(icon_data)

# Create a temporary .ico file
temp_icon = tempfile.NamedTemporaryFile(delete=False, suffix=".ico")
temp_icon.write(icon_bytes)
temp_icon.close()

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Semi-automatic pose annotator (MS COCO)")

        # Initialize YOLOv8 pose estimator
        self.pose_estimator = YOLO('yolov8x-pose.pt')

        # Frame to hold canvas and scrollbars
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=tk.YES)

        # Create scrollbars
        self.x_scroll = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL)
        self.x_scroll.grid(row=1, column=0, sticky=tk.EW)
        self.y_scroll = tk.Scrollbar(self.frame, orient=tk.VERTICAL)
        self.y_scroll.grid(row=0, column=1, sticky=tk.NS)

        self.canvas = tk.Canvas(self.frame, xscrollcommand=self.x_scroll.set, yscrollcommand=self.y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        # Configure frame to expand canvas
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Configure scrollbars to work with the canvas
        self.x_scroll.config(command=self.canvas.xview)
        self.y_scroll.config(command=self.canvas.yview)

        self.canvas.bind("<Button-1>", self.check_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.auto_keypoints = []
        self.selected_keypoint = None

        self.keypoints_order = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

        self.keypoint_connections = [
            ("left_ankle", "left_knee"),
            ("left_knee", "left_hip"),
            ("left_hip", "left_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("left_shoulder", "left_ear"),
            ("left_ear", "nose"),
            ("nose", "right_ear"),
            ("right_ear", "right_shoulder"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("right_shoulder", "right_hip"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "right_hip")
        ]

        menu = tk.Menu(root)
        root.config(menu=menu)

        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", command=self.open_image)
        file_menu.add_command(label="Save Annotations...", command=self.save_annotations)
        file_menu.add_command(label="Save Annotated Images", command=self.save_annotated_image)
        file_menu.add_command(label="Exit", command=root.quit)

        help_menu = tk.Menu(menu)
        menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Controls", command=self.show_controls)
        help_menu.add_command(label="About", command=self.show_about)

    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About")

        info = (
            "Developed by Kevin Gildea, Ph.D.\n"
            "Faculty of Engineering, LTH\n"
            "Lund University\n"
            "Email: kevin.gildea@tft.lth.se"
        )

        label = tk.Label(about_window, text=info, font=("Arial", 8))
        label.pack(pady=15)

    def show_controls(self):
        controls_window = tk.Toplevel(self.root)
        controls_window.title("Controls")

        controls_info = [
            "Left Click and Drag: Move Selected Keypoint",
            "Mouse Scroll: Navigate Image",
            "Open Image: Load a new image for annotation",
            "Save Annotations: Save current keypoints to text file",
            "Save Annotated Images: Save images with annotated keypoints"
        ]

        for info in controls_info:
            label = tk.Label(controls_window, text=info, font=("Arial", 10))
            label.pack(pady=5)

        # Close button
        close_button = tk.Button(controls_window, text="Close", command=controls_window.destroy)
        close_button.pack(pady=10)


    def open_image(self):
        self.file_path = filedialog.askopenfilename()
        if not self.file_path:
            return

        self.canvas.delete("all")
        self.image = Image.open(self.file_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Set the canvas scroll region to cover the entire image
        self.canvas.config(scrollregion=(0, 0, self.tk_image.width(), self.tk_image.height()))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        results = self.pose_estimator(self.file_path)[0]
        if results.boxes:
            self.single_person_pose = results.boxes.xyxy[0]
            if hasattr(results.keypoints, 'xy') and results.keypoints.xy.numel() > 0:
                self.auto_keypoints = []
                for xy, name in zip(results.keypoints.xy[0], self.keypoints_order):
                    x, y = int(xy[0].item()), int(xy[1].item())
                    oval = self.canvas.create_oval(x-2, y-2, x+2, y+2, fill='green')
                    label = self.canvas.create_text(x+15, y, text=name, anchor='w', font=("Arial", 8), fill='red')
                    self.auto_keypoints.append((x, y, oval, label))

    def check_click(self, event):
        self.selected_keypoint = None
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        for i, (x, y, oval, label) in enumerate(self.auto_keypoints):
            if abs(canvas_x - x) < 10 and abs(canvas_y - y) < 10:
                self.selected_keypoint = (i, oval, label)
                return

    def on_drag(self, event):
        if self.selected_keypoint:
            index, oval, label = self.selected_keypoint
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            self.canvas.coords(oval, canvas_x-2, canvas_y-2, canvas_x+2, canvas_y+2)
            self.canvas.coords(label, canvas_x + 15, canvas_y)
            self.auto_keypoints[index] = (canvas_x, canvas_y, oval, label)

    def on_release(self, event):
        self.selected_keypoint = None

    def save_annotated_image(self):
        # Create copies of the original image
        keypoints_image = self.image.copy()
        skeleton_image = self.image.copy()

        draw_keypoints = ImageDraw.Draw(keypoints_image)
        draw_skeleton = ImageDraw.Draw(skeleton_image)

        # Draw connections
        for start, end in self.keypoint_connections:
            if start in self.keypoints_order and end in self.keypoints_order:
                start_idx = self.keypoints_order.index(start)
                end_idx = self.keypoints_order.index(end)
                if start_idx < len(self.auto_keypoints) and end_idx < len(self.auto_keypoints):
                    x1, y1, _, _ = self.auto_keypoints[start_idx]
                    x2, y2, _, _ = self.auto_keypoints[end_idx]
                    draw_skeleton.line([(x1, y1), (x2, y2)], fill='green', width=10)

        # Draw keypoints
        for x, y, _, _ in self.auto_keypoints:
            s = 6
            draw_keypoints.ellipse([(x-s, y-s), (x+s, y+s)], fill='red')
            draw_skeleton.ellipse([(x-s, y-s), (x+s, y+s)], fill='red')

        # Save the annotated images
        keypoints_image_path = os.path.splitext(self.file_path)[0] + "_keypoints.png"
        skeleton_image_path = os.path.splitext(self.file_path)[0] + "_skeleton.png"
        keypoints_image.save(keypoints_image_path)
        skeleton_image.save(skeleton_image_path)

        messagebox.showinfo("Success", "Annotated images saved successfully!")

    def save_annotations(self):
        default_name = os.path.splitext(os.path.basename(self.file_path))[0] + "_keypoints.txt"
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_name, filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not file_path:
            return

        with open(file_path, 'w') as file:
            for keypoint_name, (x, y, _, _) in zip(self.keypoints_order, self.auto_keypoints):
                file.write(f"{keypoint_name}: {x}, {y}\n")

        messagebox.showinfo("Success", "Keypoints saved successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    #root.iconbitmap('ICON.ico')
    # Use the temporary .ico file
    root.iconbitmap(temp_icon.name)
    app = ImageAnnotator(root)
    # remove the temporary .ico file
    os.unlink(temp_icon.name)
    root.mainloop()
