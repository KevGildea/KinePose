# Import necessary modules
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO
import os
import cv2
import threading
from icon import icon_data
import base64
import os
import tempfile
import multiprocessing

# Convert the byte array back to bytes
icon_bytes = base64.b64decode(icon_data)

# Create a temporary .ico file
temp_icon = tempfile.NamedTemporaryFile(delete=False, suffix=".ico")
temp_icon.write(icon_bytes)
temp_icon.close()

class VideoPoseAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic pose annotator (MS COCO)")

        # Initialize YOLOv8 pose estimator
        self.pose_estimator = YOLO('yolov8x-pose.pt')

        self.keypoints_order = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", 
                                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                                "left_wrist", "right_wrist", "left_hip", "right_hip", 
                                "left_knee", "right_knee", "left_ankle", "right_ankle"]

        self.keypoint_connections = [
            ("left_ankle", "left_knee"), ("left_knee", "left_hip"), ("left_hip", "left_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("left_shoulder", "left_ear"),
            ("left_ear", "nose"), ("nose", "right_ear"), ("right_ear", "right_shoulder"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("right_shoulder", "right_hip"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("left_shoulder", "right_shoulder"),
            ("left_hip", "right_hip")
        ]

        # UI setup
        self.setup_ui()

    def setup_ui(self):
        # Step 1: Set Start and End Time
        tk.Label(self.root, text="1. Set Start and End Time").pack(pady=5)

        trim_frame = tk.Frame(self.root)
        trim_frame.pack(pady=5)
        tk.Label(trim_frame, text="Start Time (s):").pack(side=tk.LEFT)
        self.start_time_entry = tk.Entry(trim_frame, width=5)
        self.start_time_entry.pack(side=tk.LEFT)
        tk.Label(trim_frame, text="End Time (s):").pack(side=tk.LEFT)
        self.end_time_entry = tk.Entry(trim_frame, width=5)
        self.end_time_entry.pack(side=tk.LEFT)

        # Step 2: Select Video
        tk.Label(self.root, text="2. Select Video").pack(pady=5)

        open_video_button = tk.Button(self.root, text="Open Video", command=self.open_video)
        open_video_button.pack(pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=200, mode='determinate')
        self.progress_bar.pack(pady=10)

        # Exit Button
        exit_button = tk.Button(self.root, text="Exit", command=self.root.quit)
        exit_button.pack(pady=10)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if self.video_path:
            threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        start_time = float(self.start_time_entry.get() or 0)
        end_time = float(self.end_time_entry.get() or float('inf'))

        cap = cv2.VideoCapture(self.video_path)

        # Calculate frame numbers for trimming
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.progress_bar['maximum'] = end_frame - start_frame

        # Extracting video name for output directory
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(os.path.dirname(self.video_path), f"{video_name}_annotations")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        display_window = tk.Toplevel(self.root)
        display_window.title("Annotated Frames")
        display_label = tk.Label(display_window)
        display_label.pack()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated_frame = self.process_frame(pil_frame, i - start_frame, output_dir)

            self.update_display(display_label, annotated_frame)

            self.progress_bar['value'] = i - start_frame + 1
            self.root.update()

        cap.release()
        messagebox.showinfo("Completion", "Video processing completed.")
        display_window.destroy()

    def process_frame(self, image, frame_count, output_dir):
        results = self.pose_estimator(image)[0]
        keypoints_image = self.draw_pose(image, results)

        frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
        image.save(frame_path)

        annotated_frame_path = os.path.join(output_dir, f"frame_{frame_count}_annotated.png")
        keypoints_image.save(annotated_frame_path)

        keypoints_path = os.path.join(output_dir, f"frame_{frame_count}_keypoints.txt")
        self.save_keypoints(results, keypoints_path)

        return keypoints_image

    def draw_pose(self, image, results):
        keypoints_image = image.copy()
        draw = ImageDraw.Draw(keypoints_image)

        if results.boxes:
            for start, end in self.keypoint_connections:
                if start in self.keypoints_order and end in self.keypoints_order:
                    start_idx = self.keypoints_order.index(start)
                    end_idx = self.keypoints_order.index(end)
                    x1, y1 = results.keypoints.xy[0][start_idx][0:2]
                    x2, y2 = results.keypoints.xy[0][end_idx][0:2]
                    draw.line([(x1, y1), (x2, y2)], fill='green', width=10)

            for xy, name in zip(results.keypoints.xy[0], self.keypoints_order):
                x, y = int(xy[0].item()), int(xy[1].item())
                s = 6
                draw.ellipse([(x-s, y-s), (x+s, y+s)], fill='red')

        return keypoints_image

    def save_keypoints(self, results, file_path):
        with open(file_path, 'w') as file:
            for xy, name in zip(results.keypoints.xy[0], self.keypoints_order):
                x, y = int(xy[0].item()), int(xy[1].item())
                file.write(f"{name}: {x}, {y}\n")

    def update_display(self, label, image):
        tk_image = ImageTk.PhotoImage(image)
        label.config(image=tk_image)
        label.image = tk_image

if __name__ == "__main__":
    root = tk.Tk()
    #root.iconbitmap('ICON.ico')
    # Use the temporary .ico file
    root.iconbitmap(temp_icon.name)
    app = VideoPoseAnnotator(root)
    # remove the temporary .ico file
    os.unlink(temp_icon.name)
    root.mainloop()
