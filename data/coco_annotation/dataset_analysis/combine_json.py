import json

def reassign_ids(images, annotations):
    # Global counters for unique IDs
    image_id_counter = 0
    annotation_id_counter = 0

    # Assign new unique IDs to images
    for image in images:
        image['id'] = image_id_counter
        image_id_counter += 1

    # Assign new unique IDs to annotations
    for annotation in annotations:
        annotation['id'] = annotation_id_counter
        annotation_id_counter += 1
        # Update image_id in annotations to the new image ID
        # This assumes a consistent ordering of images in the combined list
        annotation['image_id'] = image_id_counter - len(images) + annotation['image_id']

def combine_and_sort_json_files_with_unique_ids(input_files, output_file):
    combined_images = []
    combined_annotations = []
    
    # Load and combine data from each file
    for file in input_files:
        with open(file, 'r') as f:
            data = json.load(f)
            combined_images.extend(data['images'])
            combined_annotations.extend(data['annotations'])

    # Assign new unique IDs to images and annotations
    reassign_ids(combined_images, combined_annotations)

    # Sort combined images and annotations
    combined_images.sort(key=lambda x: x['id'])
    combined_annotations.sort(key=lambda x: x['id'])

    # Assume categories are the same across all files and take from the first file
    combined_categories = []
    if input_files:
        with open(input_files[0], 'r') as f:
            data = json.load(f)
            combined_categories = data['categories']

    # Combine into new JSON object
    combined_data = {
        'images': combined_images,
        'annotations': combined_annotations,
        'categories': combined_categories
    }

    # Save combined and sorted data to output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    # Count images in combined JSON
    count_images_in_json(output_file)

def count_images_in_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        # return len(data['images'])
        print(f"Number of images in {file_path}: {len(data['images'])}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0

# ======================================================================
# Example usage

set_names = ['train', 'val']

for set_name in set_names:

    input_json_files = [
        f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/{set_name}_clear_simple.json', \
        f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/new_split_10_perc_train_raw/new_set/{set_name}_set_weather.json'] # Add your file paths here

    output_json_file = f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/new_additional_10_perc_split/{set_name}_all_simple_10_perc.json'

    # Count images in each JSON file
    for file in input_json_files:
        count_images_in_json(file)

    combine_and_sort_json_files_with_unique_ids(input_json_files, output_json_file)

# ======================================================================
# combine_and_sort_json_files(input_json_files, output_json_file)
# parent_dir = '/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/new_split_10_perc_train_raw'

# destination_dir = '/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/new_split_10_perc_train_raw/new_set'

# # Loop through all JSON files and count images
# import os
# from tqdm import tqdm

# # List all json files in the parent directory
# json_files = [file for file in os.listdir(parent_dir) if file.endswith(".json")]

# train_set = []
# val_set = []
# test_set = []
# # Loop through each json file
# for json_file in tqdm(json_files):
#     if "test" in json_file.split("/")[-1].split(".")[0]:
#         test_set.append(f'{parent_dir}/{json_file}')
#     elif "val" in json_file.split("/")[-1].split(".")[0]:
#         val_set.append(f'{parent_dir}/{json_file}')
#     elif "train" in json_file.split("/")[-1].split(".")[0]:
#         train_set.append(f'{parent_dir}/{json_file}')
#     else:
#         print(f"Skipping file {json_file}")
    
# # Combine all train sets, all val sets, and all test sets

# combine_and_sort_json_files_with_unique_ids(train_set, f'{destination_dir}/train_set_weather.json')
# combine_and_sort_json_files_with_unique_ids(val_set, f'{destination_dir}/val_set_weather.json')
# combine_and_sort_json_files_with_unique_ids(test_set, f'{destination_dir}/test_set_weather.json')
# ======================================================================