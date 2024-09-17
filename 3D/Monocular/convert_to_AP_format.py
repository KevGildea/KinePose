# Expected json format: https://github.com/Walter0807/MotionBERT/issues/16

import json
import os
import re

class convert2AP():
    def sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    def convert_keypoints_to_alphapose(input_folder, output_json):
        alphapose_results = []

        keypoints_order = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        keypoints_dict = {k: [0, 0, 0] for k in keypoints_order}

        for filename in sorted(os.listdir(input_folder), key=convert2AP.sort_key):
            if filename.endswith("_keypoints.txt"):
                frame_keypoints_path = os.path.join(input_folder, filename)
                image_id = filename.replace("_keypoints.txt", ".jpg") 

                with open(frame_keypoints_path, 'r') as file:
                    keypoints_data = file.readlines()

                for key in keypoints_dict:
                    keypoints_dict[key] = [0, 0, 0]

                for data in keypoints_data:
                    parts = data.strip().split(': ')
                    if parts and len(parts) == 2:
                        body_part = parts[0].replace(' ', '_').lower()
                        coords = parts[1].split(', ')
                        if body_part in keypoints_dict:
                            keypoints_dict[body_part] = [float(coords[0]), float(coords[1]), 1]

                keypoints = []
                for key in keypoints_order:
                    keypoints.extend(keypoints_dict[key])

                alphapose_results.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "score": 1 
                })

        with open(output_json, 'w') as f:
            json.dump(alphapose_results, f, indent=4)


