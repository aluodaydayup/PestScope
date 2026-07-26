import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "pest24_data"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


checkpoint = PROJECT_ROOT / "Models" / "sam2_checkpoint" / "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
predictor = SAM2ImagePredictor(sam2_model)

image_dir = DATA_ROOT / "Pest24" / "JPEGImages"
annotation_dir2 = DATA_ROOT / "Pest24" / "Annotations"

files_image = set(f.rsplit('.', 1)[0] for f in os.listdir(image_dir))
files_annotation = set(f.rsplit('.', 1)[0] for f in os.listdir(annotation_dir2))

common_files = files_image.intersection(files_annotation)

n = 15
for i in range(0, len(common_files), n):
    batch = list(common_files)[i:i + n]
    files_to_process = [(image_dir / f'{file}.jpg', annotation_dir2 / f'{file}.xml') for file in batch]

    image_batch = []
    boxes_batch = []
    total_name_list = []
    for image, annotation in files_to_process:
        image_open = Image.open(image)
        image_open = np.array(image_open.convert('RGB'))

        tree = ET.parse(annotation)
        root = tree.getroot()
        folder = root.find('folder').text
        filename = root.find('filename').text
        path = root.find('path').text
        size = root.find('size')
        width = int(size.find('width').text)
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        objects = []
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            pose = obj.find('pose').text
            truncated = int(obj.find('truncated').text)
            difficult = int(obj.find('difficult').text)

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            objects.append({
                'name': obj_name,
                'pose': pose,
                'truncated': truncated,
                'difficult': difficult,
                'bndbox': {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                }
            })
        boxes_list = []
        name_list = []
        for obj1 in objects:
            boxes_list.append([
                int(obj1['bndbox']['xmin']),
                int(obj1['bndbox']['ymin']),
                int(obj1['bndbox']['xmax']),
                int(obj1['bndbox']['ymax'])
            ])
            name_list.append(obj1['name'])
        boxes_list = np.array(boxes_list)

        total_name_list.append(name_list)
        boxes_batch.append(boxes_list)
        image_batch.append(image_open)
    predictor.set_image_batch(image_batch)
    masks_batch, scores_batch, _ = predictor.predict_batch(
        None,
        None,
        box_batch=boxes_batch,
        multimask_output=False
    )

    for file_name, pest_name_list, image_list, boxes, masks in zip(batch, total_name_list, image_batch, boxes_batch, masks_batch):
        if masks.shape == (1, 600, 800):
            masks = np.expand_dims(masks, axis=0)

        tree = ET.parse(annotation_dir2 / f'{file_name}.xml')
        root = tree.getroot()
        count = 0
        for pest_name, box, mask in zip(pest_name_list, boxes.tolist(), masks):
            for i, object_element in enumerate(root.findall('object')):
                object_name = object_element.find('name').text
                bndbox_element = object_element.find('bndbox')
                xmin = int(bndbox_element.find('xmin').text)
                ymin = int(bndbox_element.find('ymin').text)
                xmax = int(bndbox_element.find('xmax').text)
                ymax = int(bndbox_element.find('ymax').text)
                bndbox = [xmin, ymin, xmax, ymax]

                if str(object_name) == str(pest_name) and str(box) == str(bndbox):
                    if mask.shape == (1, 600, 800):
                        mask = mask.squeeze(0)

                    mask_image = mask * 255
                    mask_image = Image.fromarray(mask_image)
                    mask_image = mask_image.convert('L')
                    save_path = Path("Pest24") / "mask_image" / f'mask_image_{file_name}_{pest_name}_{str(box)}.png'
                    save_path_mask_file = DATA_ROOT / save_path
                    mask_image.save(save_path_mask_file)

                    mask_element = ET.Element('mask_path')
                    mask_element.text = save_path.as_posix()
                    object_element.append(mask_element)

                    tree.write(annotation_dir2 / f'{file_name}.xml', encoding='utf-8', xml_declaration=True)
