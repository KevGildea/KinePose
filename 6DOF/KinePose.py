# Import necessary modules
import tkinter as tk
from tkinter import filedialog, Label, Entry, messagebox, simpledialog
import tkinter.font as tkFont
import numpy as np
import pandas as pd
import math
import copy
from kinematics import kinematics as kin
from visualisations import visualisations as vis
from utilities import utilities as utils
from utilities import StdoutRedirector
import sys
import time
import threading
from icon import icon_data
import base64
import os
import tempfile
import ast

# convert the byte array back to bytes
icon_bytes = base64.b64decode(icon_data)

# create a temporary .ico file
temp_icon = tempfile.NamedTemporaryFile(delete=False, suffix=".ico")
temp_icon.write(icon_bytes)
temp_icon.close()

# initialize Tkinter root
root = tk.Tk()
root.title("KinePose 3D-6DOF")

# create menu bar
menu_bar = tk.Menu(root)

# add menu bar to the root window
root.config(menu=menu_bar)

# about me :)
def show_about():
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

# add 'about' to menu bar
menu_bar.add_command(label="About", command=show_about)


# Variables
data = None
prediction = None
dir_graph = None
chain = None
pos_as = None
ori_as = None
results_of_minimizations = None
poses = None
EAs = None
optimization_results = None

# Variables to store user input for frame selection in optimisation
optimization_start_frame = tk.StringVar()
optimization_stop_frame = tk.StringVar()

# Temporary weight value for optimisation
temp_weight_var = tk.StringVar(value="0")

# process kinematic chain data
def process_kinematic_chain(file_path):
    global data, dir_graph, chain
    data = pd.read_excel(file_path, engine='openpyxl')

    # converting column data to list
    ori_global = data['Global orientation'].tolist()
    ori_global = [np.array(np.matrix(i).reshape((3, 3))) for i in ori_global]
    ori_global = [i.T for i in ori_global]
    pos_global = data['Global position'].tolist()
    pos_global = [np.array(np.matrix(i)).reshape((1, 3))[0] for i in pos_global]

    # show an open file dialog and get the selected file path
    file_path = filedialog.askopenfilename(title="Select a directed graph file", filetypes=[("Text Files", "*.txt")])

    # check if a file was selected
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
            dir_graph = ast.literal_eval(file_content)
    else:
        print("No file was selected.")

    # reformulate kinematic chain
    chain = kin.Rev_FK_MDH(ori_global, pos_global, dir_graph)

    # plot the kinematic chain using FK
    ori_a, pos_a, pos_a_rel = kin.FK_MDH(copy.deepcopy(chain), dir_graph)
    vis.plot_chain_global(pos_a, ori_a, dir_graph, title='')

# process 3D keypoints data
def process_3d_keypoints(file_path):
    global prediction, poses, dir_graph_pose
    prediction = np.loadtxt(file_path)
    prediction = prediction.reshape(int(len(prediction)/(17)), 17, 3)

    # set hip to 0,0,0 in the first frame
    prediction = utils.root2origin(prediction)

    # remove 'Neck/Nose' from procedure
    prediction = np.delete(prediction, 9, axis=1)

    poses = prediction

    dir_graph_pose = {0: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                        1: [2,3],
                        2: [3],
                        4: [5,6],
                        5: [6],
                        7: [8,9,10,11,12,13,14,15],
                        8: [9],
                        10: [11,12],
                        11: [12],
                        13: [14,15],
                        14: [15]}

    # plot frames of the 3D pose estimate
    vis.plot_pose_local_aniplusslider(prediction, dir_graph_pose, start=0, stop=len(prediction), title='')

# load kinematic chain file
def load_kinematic_chain():
    file_path = filedialog.askopenfilename(title="Select Kinematic Chain File", filetypes=[("Excel Files", "*.xlsx")])
    if file_path:
        try:
            process_kinematic_chain(file_path)
            print("----------------------------------------------------------------------------------------------------------------")
            print(f"Kinematic chain file loaded: \n {file_path}")
            print("----------------------------------------------------------------------------------------------------------------")
            print()
        except Exception as e:
            print(f"Error loading kinematic chain file: {e}")
            print()

# load 3D keypoints file
def load_3d_keypoints():
    file_path = filedialog.askopenfilename(title="Select 3D Keypoints File", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            process_3d_keypoints(file_path)
            print("----------------------------------------------------------------------------------------------------------------")
            print(f"3D keypoints file loaded: \n {file_path}")
            print("----------------------------------------------------------------------------------------------------------------")
            print()
        except Exception as e:
            print(f"Error loading 3D keypoints file: {e}")
            print()

def process_ROM_data(entry):
    if entry == "FULL":
        # convert "FULL" into the specified list of tuples
        return [(-1, 1), (-1, 1), (-1, 1)]
    elif entry == "NONE":
        # return the list of zeros for "NONE"
        return [(0,0), (0,0), (0,0)]
    else:
        # process a list of tuples, dividing each value by 2*math.pi and convert to list
        try:
            tuples_list = ast.literal_eval(entry)  # convert string to list of tuples
            return [(round(a / (2 * math.pi), 3), round(b / (2 * math.pi), 3)) for a, b in tuples_list]
        except (ValueError, SyntaxError):
            # handle the case where conversion is not possible
            return "Error processing entry: " + entry

def pre_run_setup():
    global EAs

    global bds_new, vector_pairs, dir_graph_pose

    global fps

    file_path = filedialog.askopenfilename(title="Select the vector pairs to be used in IK", filetypes=[("Text Files", "*.txt")])

    # check if a file was selected
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
            vector_pairs = ast.literal_eval(file_content)
    else:
        print("No file was selected.")
    
    # prompt for FPS
    fps = simpledialog.askfloat("Input", "Enter the frames per second (FPS) of the video:", minvalue=0.1, maxvalue=120.0)

    if fps is None:
        print("FPS value not provided. Operation cancelled.")
        print()
        return

    data_ROMs = data['ROMs'].tolist()
    bds_new = [process_ROM_data(entry) for entry in data_ROMs]

    # ask user whether to use manual positioning
    use_manual_positioning = messagebox.askyesno("Manual Positioning", "Select yes for plot of joint indices, and optional manual repositioning for the initial frame.")
    if use_manual_positioning:
        bds_plot_new = [[[value * 2 * math.pi for value in pair] for pair in sublist] for sublist in bds_new]
        EAs = vis.plot_chain_interactive(chain, dir_graph, dir_graph_pose, bds_plot_new, poses, title='Optional manual repositioning for initial frame, and joint indices', vector_pairs=vector_pairs)
        EAs = [ea / (2 * math.pi) for ea in EAs]
    else:
        EAs = [0] * len(chain)*3 # default values if manual positioning is not used

    if not EAs: 
        EAs = [0] * len(chain) * 3

def start_analysis_thread():
    pre_run_setup()
    analysis_thread = threading.Thread(target=run_analysis)
    analysis_thread.start()

def update_plot():
    vis.plot_chain_global_frames_aniplusslider(pos_as, ori_as, dir_graph, 0, len(results_of_minimizations), title='')

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
            angular_velocity = (log_so3(R_next) - log_so3(R_prev)) / (2 * delta_t)
            
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



# function to run IK
def run_analysis():
    global chain, pos_as, ori_as, results_of_minimizations, poses

    global ik_results_loaded_from_file

    global EAs

    global bds_new, vector_pairs

    global fps
    
    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if data is None or prediction is None:
        print("Please load both kinematic chain and keypoints files.")
        print()
        return

    try:
        optimization_start = int(optimization_start_frame.get()) if optimization_start_frame.get() else 0
        optimization_stop = int(optimization_stop_frame.get()) if optimization_stop_frame.get() else len(prediction)
    except ValueError:
        print("Invalid frame numbers entered. Please enter valid integers.")
        print()
        return

    # frames to optimize
    poses = prediction[optimization_start:optimization_stop + 1]
    
    try:
        temp_weight = float(temp_weight_var.get())
        if not (0 <= temp_weight <= 5):
            print("Invalid temporal weight entered. Please enter a valid number between 0 and 5.")
            print()
            return
    except ValueError:
        print("Invalid temporal weight entered. Please enter a valid number.")
        print()
        return

    start_time = time.time()

    frame_based_results = None
    if float(temp_weight_var.get()) != 0:
        print("----------------------------------------------------------------------------------------------------------------")
        print("Performing frame-based inverse kinematics pre-optimisation")
        print("----------------------------------------------------------------------------------------------------------------")
        print()
    else:
        print("----------------------------------------------------------------------------------------------------------------")
        print("Performing frame-based inverse kinematics optimisation (\u03BB = 0)")
        print("----------------------------------------------------------------------------------------------------------------")
        print()

    bds_new = [tuple_item for sublist in bds_new for tuple_item in sublist]
    frame_based_results = kin.IK_opt_frames(chain, poses, dir_graph, bds_new, EAs,vector_pairs)

    if float(temp_weight_var.get()) != 0:
        print("----------------------------------------------------------------------------------------------------------------")
        print("Performing temporal inverse kinematics optimisation")
        print("----------------------------------------------------------------------------------------------------------------")
        print()
        if frame_based_results:
            initial_weights = np.array([res.x for res in frame_based_results]).flatten()
        else:
            initial_weights = [0] * len(chain)*3 * len(poses)
        
        bds_new = bds_new * len(poses)
        results_of_minimizations = kin.IK_opt_frames_temporal(chain, poses, dir_graph, bds_new, temp_weight, initial_weights,vector_pairs)
        results_of_minimizations = np.reshape(results_of_minimizations.x, (len(poses), len(chain)*3))

    else:
        results_of_minimizations = np.array([res.x for res in frame_based_results]).flatten()
        results_of_minimizations = np.reshape(results_of_minimizations, (len(poses), len(chain)*3))
    
    end_time = time.time()
    time_taken = end_time - start_time

    hours = int(time_taken // 3600)
    minutes = int((time_taken % 3600) // 60)
    seconds = time_taken % 60

    print("Analysis complete.")
    print()
    print(f"Time taken for optimisation: {hours} hours, {minutes} minutes, {seconds:.2f} seconds.")
    print()

    ## Save the 3D predictions to text file
    #with open("IKpose.txt", "w") as a_file:
    #    for row in results_of_minimizations:
    #        np.savetxt(a_file, row)

    # Load IKpose frames weights from previous optimisation
    #results_of_minimizations = np.loadtxt("IKpose.txt").reshape(len(poses), len(chain)*3)

    results_of_minimizations = utils.smooth_weights(results_of_minimizations, max_window_length=20, polyorder=2)

    # display optimised and smoothed motion
    pos_as = []
    ori_as = []
    chain_reoris =[]
    for frame in range(len(results_of_minimizations)):
        EAs = (results_of_minimizations[frame]) * 2 * math.pi
        EAs = np.reshape(EAs,(len(chain),3))
        chain_reori = kin.FK_MDH_reori(copy.deepcopy(chain), EAs)
        ori_a, pos_a, _ = kin.FK_MDH(chain_reori, dir_graph)
        pos_as.append(pos_a)
        ori_as.append(ori_a)
        chain_reoris.append(chain_reori)

    smoothed_local_dofs, smoothed_global_dofs, smoothed_global_poss = chain_reoris, ori_as, pos_as

    # save results
    save_transformation_data_to_excel(np.array(smoothed_local_dofs)[:, :, :3, :3], np.array(smoothed_local_dofs)[:, :, :3, 3], "results/local_dofs.xlsx")
    save_transformation_data_to_excel(np.array(smoothed_global_dofs), np.array(smoothed_global_poss), "results/global_dofs.xlsx")

    if len(results_of_minimizations) > 1:
        angular_velocities = calculate_angular_velocities(np.array(smoothed_local_dofs)[:, :, :3, :3], 1/fps)
        save_angular_velocities_to_excel(angular_velocities, "results/angular_velocities.xlsx")

    ik_results_loaded_from_file = False
    root.after(0, update_plot)



# layout
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
pad_x = 10
pad_y = 5

# widgets
tk.Button(root, text="1. Load kinematic chain", command=load_kinematic_chain).grid(row=0, column=0, columnspan=2, sticky='ew', padx=pad_x, pady=pad_y)
tk.Button(root, text="2. Load 3D pose", command=load_3d_keypoints).grid(row=1, column=0, columnspan=2, sticky='ew', padx=pad_x, pady=pad_y)
Label(root, text="Start frame:").grid(row=3, column=0, sticky='w', padx=pad_x, pady=pad_y)
Entry(root, textvariable=optimization_start_frame).grid(row=3, column=1, sticky='ew', padx=pad_x, pady=pad_y)
Label(root, text="Stop frame:").grid(row=4, column=0, sticky='w', padx=pad_x, pady=pad_y)
Entry(root, textvariable=optimization_stop_frame).grid(row=4, column=1, sticky='ew', padx=pad_x, pady=pad_y)
Label(root, text="Temporal weight (\u03BB):").grid(row=5, column=0, sticky='w', padx=pad_x, pady=pad_y)
Entry(root, textvariable=temp_weight_var).grid(row=5, column=1, sticky='ew', padx=pad_x, pady=pad_y)
tk.Button(root, text="3. Run IK", command=start_analysis_thread).grid(row=6, column=0, columnspan=2, sticky='ew', padx=pad_x, pady=pad_y)

output_font = tkFont.Font(family="Helvetica", size=8) 
output_text = tk.Text(root, height=10, font=output_font)
output_text.grid(row=8, column=0, columnspan=2, sticky='ew', padx=pad_x, pady=pad_y)

# redirect stdout and stderr
sys.stdout = StdoutRedirector(output_text)
sys.stderr = StdoutRedirector(output_text)

#root.iconbitmap('ICON.ico')

# use temporary .ico file
root.iconbitmap(temp_icon.name)

# remove the temporary .ico file
os.unlink(temp_icon.name)

# Start event loop
root.mainloop()
