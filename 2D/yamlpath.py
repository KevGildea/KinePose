import ultralytics
import os

# Get the directory of the ultralytics package
ultralytics_dir = os.path.dirname(ultralytics.__file__)

# Construct the path to the default.yaml file
yaml_file_path = os.path.join(ultralytics_dir, 'cfg', 'default.yaml')

print("The path to the default.yaml file is:", yaml_file_path)
