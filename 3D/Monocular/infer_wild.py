import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save, render_and_save_image
from collections import OrderedDict # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/23
from convert_to_AP_format import convert2AP as convert2AP
import multiprocessing
import tkinter as tk
from tkinter import filedialog

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="MB_ft_h36m.yaml", help="Path to the config file.") #MB_ft_h36m_global_lite.yaml
    parser.add_argument('-e', '--evaluate', default='best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-p', '--pose_folder_path', type=str, help='2D poses obtained from custom tools')    
    parser.add_argument('-j', '--json_path', default='results-AP_format.json', type=str, help='alphapose detection result json path') 
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', default='Results', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

def get_user_paths():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    pose_folder_path = filedialog.askdirectory(title="Select Pose Folder Path")
    vid_path = filedialog.askopenfilename(title="Select Video File")

    return pose_folder_path, vid_path

def is_image_file(filepath):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    return any(filepath.lower().endswith(ext) for ext in image_extensions)

def is_video_file(filepath):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return any(filepath.lower().endswith(ext) for ext in video_extensions)

def rotate_points(points, angle, axis='x'):
    """Rotate points by a given angle around a specified axis."""
    angle = np.radians(angle)
    if axis == 'x':
        rot_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rot_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    else:  # axis == 'z'
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    return np.dot(points, rot_matrix)

def main(opts):
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    pose_folder_path, vid_path = get_user_paths()

    #with open('selected_paths.txt', 'r') as file:
    #    pose_folder_path = file.readline().strip()
    #    vid_path = file.readline().strip()
    #pose_folder_path = input("Enter the path to the folder containing 2D poses: ")
    #vid_path = input("Enter the path to the video file: ")
    #pose_folder_path = 'C:/Users/ke4446gi/Work Folders/Desktop/Journal of biomechanics/Github repo/upload/3D pose/MotionBERT/baseball_annotations'
    #vid_path = 'C:/Users/ke4446gi/Work Folders/Desktop/Journal of biomechanics/Github repo/upload/3D pose/MotionBERT/Example video/baseball.mp4'

    #opts = parse_args()
    args = get_config(opts.config)

    convert2AP.convert_keypoints_to_alphapose(pose_folder_path, 'results-AP_format.json')

    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', 'best_epoch.bin')
    checkpoint = torch.load('best_epoch.bin', map_location=lambda storage, loc: storage)

    # Block to handle the 'module.' prefix issue: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/23
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_pos'].items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    checkpoint['model_pos'] = new_state_dict

    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()
    #testloader_params = {
    #        'batch_size': 1,
    #        'shuffle': False,
    #        'num_workers': 8,
    #        'pin_memory': True,
    #        'prefetch_factor': 4,
    #        'persistent_workers': True,
    #        'drop_last': False
    #}
    testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0,  # No additional workers for loading data
        'pin_memory': True,
        'persistent_workers': False,  # Not relevant when num_workers is 0
        'drop_last': False
    }

    #vid = imageio.get_reader(vid_path,  'ffmpeg')
    #fps_in = vid.get_meta_data()['fps']
    #vid_size = vid.get_meta_data()['size']
    #os.makedirs(opts.out_path, exist_ok=True)

    # Determine if the file is an image or a video
    if is_image_file(vid_path):
        # Process as an image
        img = imageio.imread(vid_path)
        # [Add your image processing code here]
    elif is_video_file(vid_path):
        # Existing video processing code
        vid = imageio.get_reader(vid_path,  'ffmpeg')
        fps_in = vid.get_meta_data()['fps']
        vid_size = vid.get_meta_data()['size']
        # [Rest of your existing video processing code]
    else:
        print("Unsupported file format.")

    if opts.pixel:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)

    if is_video_file(vid_path):
        render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
    else:
        render_and_save_image(results_all, '%s/X3D.png' % (opts.out_path))

    if opts.pixel:
        # Convert to pixel coordinates
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
    np.save('%s/X3D.npy' % (opts.out_path), results_all)

    #save in the format required for KinePose
    keypoints_file_path = os.path.join(opts.out_path, '3D_keypoints.txt')
    with open(keypoints_file_path, 'w') as file:
        for frame in results_all:
            frame = rotate_points(frame, 90, axis='x')
            for keypoint in frame:
                file.write(f"{keypoint[0]} {keypoint[1]} {keypoint[2]}\n")

if __name__ == '__main__':
    opts = parse_args()
    main(opts)