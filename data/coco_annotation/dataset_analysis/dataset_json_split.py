import json
import random
import os

def split_and_sort_json_data(input_file, train_file, val_file, test_file, val_percent=0.05, train_percent=0.1, seed=42):
    # Set seed for reproducibility
    random.seed(seed)

    # Load JSON data from the file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Extract the 'images' list and shuffle
    images = data['images']
    random.shuffle(images)

    # Calculate split sizes
    total_images = len(images)
    num_val = int(total_images * val_percent)
    num_train = int(total_images * train_percent)
    num_test = total_images - num_val - num_train

    # Split images into train, validation, and test sets
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # Function to filter, sort images and annotations for a given set of images
    def prepare_data(selected_images):
        selected_images.sort(key=lambda x: x['id'])
        selected_image_ids = set([image['id'] for image in selected_images])
        selected_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] in selected_image_ids]
        selected_annotations.sort(key=lambda x: x['image_id'])
        return selected_images, selected_annotations

    # Prepare and save data for each set
    for image_set, filename in [(train_images, train_file), (val_images, val_file), (test_images, test_file)]:
        sorted_images, sorted_annotations = prepare_data(image_set)
        new_data = {
            'images': sorted_images,
            'annotations': sorted_annotations,
            'categories': data['categories']
        }
        with open(filename, 'w') as file:
            json.dump(new_data, file, indent=4)

def check_dir_exists_and_create(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Example usage
# input_json_file = f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/dense_fog_night.json'
# train_json_file = f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/dataset_analysis/{input_json_file.split("/")[-1].split(".")[0]}_train_set.json'
# val_json_file = f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/dataset_analysis/{input_json_file.split("/")[-1].split(".")[0]}_val_set.json'
# test_json_file = f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/dataset_analysis/{input_json_file.split("/")[-1].split(".")[0]}_test_set.json'
# val_percentage = 0.05
# test_percentage = 0.1
# split_and_sort_json_data(input_json_file, train_json_file, val_json_file, test_json_file, val_percent=val_percentage, test_percent=test_percentage)

# Loop over all json files and split them
parent_dir = "/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation"

dest_dir = "/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/new_split_50_perc_train_raw"

check_dir_exists_and_create(parent_dir)
check_dir_exists_and_create(dest_dir)

# list all json files in the parent directory
import os
from tqdm import tqdm

# List all json files in the parent directory
json_files = [file for file in os.listdir(parent_dir) if file.endswith(".json")]

cnt = 0
for json_file in tqdm(json_files):
    # Skip simple json files
    if "simple" in json_file.split("/")[-1].split(".")[0] or "val" in json_file.split("/")[-1].split(".")[0] or "test" in json_file.split("/")[-1].split(".")[0] or "train" in json_file.split("/")[-1].split(".")[0]:
        print(f'Skipping simple json file: {json_file}')
        continue
    input_json_file = f'{parent_dir}/{json_file}'
    train_json_file = f'{dest_dir}/{json_file.split("/")[-1].split(".")[0]}_train_set.json'
    val_json_file = f'{dest_dir}/{json_file.split("/")[-1].split(".")[0]}_val_set.json'
    test_json_file = f'{dest_dir}/{json_file.split("/")[-1].split(".")[0]}_test_set.json'
    
    train_percentage = 0.5
    val_percentage = 0.05
    
    split_and_sort_json_data(input_json_file, train_json_file, val_json_file, test_json_file, val_percent=val_percentage, train_percent=train_percentage)
    cnt += 1

print(f'Number of json files processed: {cnt}')