import json
import matplotlib.pyplot as plt

# Sample data in COCO format
# data = {
#     "images": [
#         # Sample image data
#     ],
#     "annotations": [
#         {"image_id": 0, "id": 0, "category_id": 0, "bbox": [1343.57, 479.52, 372.45, 245.68], "area": 91503.516, "iscrowd": 0},
#         {"image_id": 0, "id": 1, "category_id": 0, "bbox": [1290.73, 490.83, 229.27, 149.79], "area": 34342.353, "iscrowd": 0},
#         # Other annotations...
#         {"image_id": 4, "id": 21, "category_id": 1, "bbox": [208.94, 444.64, 77.05, 278.17], "area": 21432.999, "iscrowd": 0},
#         {"image_id": 4, "id": 22, "category_id": 1, "bbox": [919.2, 421.6, 211.02, 358.85], "area": 75724.527, "iscrowd": 0}
#     ],
#     "categories": [
#         {"supercategory": "Vehicle", "id": 0, "name": "Vehicle"},
#         {"supercategory": "Pedestrian", "id": 1, "name": "Pedestrian"}
#     ]
# }

# Read the JSON file
import json

file_name = '/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/hrfuser/data/dense_pkl_files/coco_annotations/dense_infos_all_coco.json'

# Read the JSON file
with open(file_name, 'r') as f:
    data = json.load(f)

# Parsing the JSON data
annotations = data['annotations']
categories = {cat['id']: cat['name'] for cat in data['categories']}

# Organizing the areas by category
areas_by_category = {cat_name: [] for cat_name in categories.values()}
for ann in annotations:
    category_name = categories[ann['category_id']]
    areas_by_category[category_name].append(ann['area'])

# Plotting histograms
fig, axs = plt.subplots(len(areas_by_category), figsize=(10, 5 * len(areas_by_category)))
if len(areas_by_category) == 1:
    axs = [axs]  # Making sure axs is always a list

for ax, (category, areas) in zip(axs, areas_by_category.items()):
    ax.hist(areas, edgecolor='black')
    ax.set_title(f'Area Distribution for {category}')
    ax.set_xlabel('Area')
    ax.set_ylabel('Frequency')
    # grid
    ax.grid(True)
    # y axis in log scale
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig(f'/home/kpatel2s/kpatel2s/sensor_fusion_rnd/KevinPatelRnD/mt_detr_cuda11p1/data/coco_annotation/dataset_analysis/{file_name.split("/")[-1].split(".")[0]}_area_distribution.png')
