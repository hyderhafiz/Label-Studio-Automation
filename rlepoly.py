import json
import os
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob

################################################ Add Class here ####################################
class_map = {   
    'Ambulance': 0,
    'Car': 1,
    'Bus': 2,
    'Truck': 3,
    'Motorcycle': 4
}

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i: self.i + size]
        self.i += size
        return int(out, 2)

def access_bit(data, num):
    """From bytes array to bits by num position."""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift

def bytes2bit(data):
    """Get bit string from bytes data."""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])

def decode_rle(rle, print_params: bool = False):
    """Convert LS RLE to numpy uint8 3D image [width, height, channel]."""
    input_stream = InputStream(bytes2bit(rle))
    num = input_stream.read(32)
    word_size = input_stream.read(5) + 1
    rle_sizes = [input_stream.read(4) + 1 for _ in range(4)]
    if print_params:
        print("RLE params:", num, word_size, rle_sizes)
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input_stream.read(1)
        j = i + 1 + input_stream.read(rle_sizes[input_stream.read(2)])
        if x:
            val = input_stream.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input_stream.read(word_size)
                out[i] = val
                i += 1
    return out

def rle_to_mask(rle, width, height):
    mask = decode_rle(rle)
    mask = np.reshape(mask, [height, width, 4])[:, :, 3]
    cv2.imwrite('mask.png', mask)
    return mask

def separate_instances(mask, min_instance_area=100):
    num_labels, labeled_img = cv2.connectedComponents(mask.astype(np.uint8))
    instances = []
    areas = np.bincount(labeled_img.flatten())[1:]
    for label, area in enumerate(areas, start=1):
        if area < min_instance_area:
            continue  # Skip small instances
        instance = (labeled_img == label).astype(np.uint8)
        instances.append(instance)
    return instances

def mask_to_polygon(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    polygon = []
    for contour in contours:
        contour = contour.flatten().tolist()
        contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
        polygon.extend(contour_pairs)
    return polygon

def convert_to_yolo(dataset_path, image_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    # If dataset is a single dict, wrap it in a list.
    if isinstance(dataset, dict):
        dataset = [dataset]
    for item in tqdm(dataset, desc="Processing JSON"):
        # Get image URL from task data
        image_in_json = item.get('task', {}).get('data', {}).get('image')
        if not image_in_json:
            print("No image found in task data.")
            continue

        base_name = os.path.basename(image_in_json)
        image_filename = os.path.join(image_path, base_name)
        if not os.path.exists(image_filename):
            # If the file is not found, try removing a prefix before a dash if present.
            if '-' in base_name:
                new_base_name = base_name.split('-', 1)[1]
                image_filename = os.path.join(image_path, new_base_name)
            if not os.path.exists(image_filename):
                print(f"Image file not found: {image_filename}")
                continue

        image = cv2.imread(image_filename)
        if image is None:
            print(f"Failed to read image: {image_filename}")
            continue
        image_height, image_width = image.shape[:2]
        annotations = []

        # Use 'annotation' key instead of 'annotations'
        if 'annotation' in item and 'result' in item['annotation'] and len(item['annotation']['result']) > 0:
            for annotation in item['annotation']['result']:
                try:
                    if annotation['type'] == 'polygonlabels':
                        label = annotation['value']['polygonlabels'][0]
                        points = annotation['value']['points']
                        normalized_points = [(point[0] / 100.0, point[1] / 100.0) for point in points]
                        flattened_points = [coord for point in normalized_points for coord in point]
                        annotations.append([label] + flattened_points)
                    elif annotation['type'] == 'brushlabels':
                        label = annotation['value']['brushlabels'][0]
                        rle = annotation['value']['rle']
                        mask = rle_to_mask(rle, image_width, image_height)
                        if mask is None:
                            continue
                        instances = separate_instances(mask)
                        for instance in instances:
                            polygon = mask_to_polygon(instance)
                            if len(polygon) > 5:
                                normalized_polygon = [(point[0] / image_width, point[1] / image_height) for point in polygon]
                                flattened_polygon = [coord for point in normalized_polygon for coord in point]
                                annotations.append([label] + flattened_polygon)
                except Exception as e:
                    print("Error processing annotation:", e)
                    continue

            # Write annotations to YOLO format file if any annotations were found.
            if annotations:
                yolo_filename = os.path.splitext(image_filename)[0].replace("images", "labels") + '.txt'
                os.makedirs(os.path.dirname(yolo_filename), exist_ok=True)
                print(f"Writing YOLO file: {yolo_filename}")
                with open(yolo_filename, 'w') as f:
                    for ann in annotations:
                        try:
                            class_name = ann[0]
                            coordinates = ann[1:]
                            if class_name == 'r':
                                class_name = 'other'
                            line = f"{class_map[class_name]} {' '.join(map(str, coordinates))}\n"
                            f.write(line)
                        except Exception as e:
                            print("Error writing annotation:", e)
                            continue


base_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(base_dir, "data/json")
image_dir = os.path.join(base_dir, "data/images")
txt_dir = os.path.join(base_dir, "data/labels")
os.makedirs(txt_dir, exist_ok=True)

for json_file in tqdm(glob(os.path.join(json_dir, "*.json")), desc="Converting JSON files"):
    convert_to_yolo(json_file, image_dir)
