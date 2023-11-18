import json

def verify_unique_ids(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Extract image and annotation IDs
    image_ids = {image['id'] for image in data['images']}
    annotation_ids = {annotation['id'] for annotation in data['annotations']}

    # Check for duplicates
    if len(image_ids) != len(data['images']):
        print("Duplicate image IDs found.")
        return False
    if len(annotation_ids) != len(data['annotations']):
        print("Duplicate annotation IDs found.")
        return False

    print("All image and annotation IDs are unique.")
    return True

# Example usage
json_file = '/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/new_additional_50_perc_split/train_all_simple_50_perc.json'  # Replace with your file path
verify_unique_ids(json_file)
