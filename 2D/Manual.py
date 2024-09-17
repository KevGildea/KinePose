import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
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
        self.root.title("Manual pose annotator (MS COCO)")

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

        self.canvas.bind("<Button-1>", self.annotate_pixel)
        self.canvas.bind("<Button-3>", self.undo_annotation)

        self.annotated_pixels = []
        self.keypoints_order = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        self.current_keypoint_index = 0

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

        # Label to display the current keypoint being annotated
        self.current_keypoint_label = tk.Label(root, text=f"Annotate: {self.keypoints_order[self.current_keypoint_index]}", font=("Arial", 12))
        self.current_keypoint_label.pack(side=tk.BOTTOM, pady=10)

    def open_image(self):
        self.file_path = filedialog.askopenfilename()
        if not self.file_path:
            return

        # Reset the canvas and annotations
        self.canvas.delete("all")
        self.annotated_pixels = []
        self.current_keypoint_index = 0
        self.update_keypoint_label()

        self.image = Image.open(self.file_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Configure the canvas scroll region to the size of the image
        self.canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def annotate_pixel(self, event):
        if self.current_keypoint_index >= len(self.keypoints_order):
            return

        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        keypoint_name = self.keypoints_order[self.current_keypoint_index]
        oval = self.canvas.create_oval(x-2, y-2, x+2, y+2, fill='red')
        text = self.canvas.create_text(x, y-10, text=keypoint_name, fill='blue')
        self.annotated_pixels.append((x, y, oval, text))
        self.current_keypoint_index += 1
        self.update_keypoint_label()

    def undo_annotation(self, event):
        if self.annotated_pixels:
            _, _, oval, text = self.annotated_pixels.pop()
            self.canvas.delete(oval)
            self.canvas.delete(text)
            self.current_keypoint_index -= 1
            self.update_keypoint_label()

    def update_keypoint_label(self):
        if self.current_keypoint_index < len(self.keypoints_order):
            self.current_keypoint_label.config(text=f"Annotate: {self.keypoints_order[self.current_keypoint_index]}")
        else:
            self.current_keypoint_label.config(text="All keypoints annotated!")

    def save_annotated_image(self):
        # Create copies of the original image
        keypoints_image = self.image.copy()
        skeleton_image = self.image.copy()
        only_skeleton_image = self.image.copy()

        draw_keypoints = ImageDraw.Draw(keypoints_image)
        draw_skeleton = ImageDraw.Draw(skeleton_image)
        draw_only_skeleton = ImageDraw.Draw(only_skeleton_image)

        # Draw each annotated keypoint on both images
        for idx, (x, y, _, _) in enumerate(self.annotated_pixels):
            keypoint_name = self.keypoints_order[idx]
            draw_keypoints.ellipse([(x-2, y-2), (x+2, y+2)], fill='red')
            draw_keypoints.text((x, y-10), keypoint_name, fill='blue')
            
            draw_skeleton.ellipse([(x-2, y-2), (x+2, y+2)], fill='red')
            draw_skeleton.text((x, y-10), keypoint_name, fill='blue')

        # Draw connections between keypoints on the skeleton images
        for start, end in self.keypoint_connections:
            if start in self.keypoints_order[:len(self.annotated_pixels)] and end in self.keypoints_order[:len(self.annotated_pixels)]:
                start_idx = self.keypoints_order.index(start)
                end_idx = self.keypoints_order.index(end)
                x1, y1 = self.annotated_pixels[start_idx][:2]
                x2, y2 = self.annotated_pixels[end_idx][:2]
                draw_skeleton.line([(x1, y1), (x2, y2)], fill='green', width=2)
                draw_only_skeleton.line([(x1, y1), (x2, y2)], fill='green', width=2)

        # Save the annotated images
        keypoints_image_path = os.path.splitext(self.file_path)[0] + "_keypoints.png"
        skeleton_image_path = os.path.splitext(self.file_path)[0] + "_skeleton+keypoints.png"
        only_skeleton_image_path = os.path.splitext(self.file_path)[0] + "_skeleton.png"
        keypoints_image.save(keypoints_image_path)
        skeleton_image.save(skeleton_image_path)
        only_skeleton_image.save(only_skeleton_image_path)

    def save_annotations(self):
        default_name = os.path.splitext(os.path.basename(self.file_path))[0] + ".txt"
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_name, filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not file_path:
            return

        with open(file_path, 'w') as file:
            for idx, (x, y, _, _) in enumerate(self.annotated_pixels):
                keypoint_name = self.keypoints_order[idx]
                file.write(f"{keypoint_name}: {x}, {y}\n")

        messagebox.showinfo("Success", "Annotations saved successfully!")

    def show_controls(self):
        controls_window = tk.Toplevel(self.root)
        controls_window.title("Controls")

        controls = [
            ("Left Click", "Annotate the current keypoint"),
            ("Right Click", "Undo the last annotation"),
            ("File -> Open Image", "Open an image for annotation"),
            ("File -> Save Annotations", "Save the annotated keypoints to a text file")
        ]

        for control, description in controls:
            label = tk.Label(controls_window, text=f"{control}: {description}", font=("Arial", 8))
            label.pack(pady=5)

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

if __name__ == "__main__":
    root = tk.Tk()
    #root.iconbitmap('ICON.ico')
    # Use the temporary .ico file
    root.iconbitmap(temp_icon.name)
    app = ImageAnnotator(root)
    # remove the temporary .ico file
    os.unlink(temp_icon.name)
    root.mainloop()
