from typing import Tuple, Union, Generator
import cv2
import numpy as np
from ultralyticsplus import YOLO, render_result
from PIL import Image
from torchvision.ops import nms
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
import matplotlib.patches as mpatches


class PatchDetection:
    def __init__(self, x, y, res, img,model):
        self.x = x
        self.y = y
        self.res = res
        self.boxes=res.boxes
        self.n_box = len(self.boxes)
        self.img = img
        self.model=model

    def __str__(self):
        return (f"PatchDetection(x={self.x}, y={self.y}, n_box={self.n_box}, "
                f"img_shape={self.img.shape if hasattr(self.img, 'shape') else 'unknown'})")


    def display_image(self):
        plt.imshow(self.img)
        plt.axis('off')
        plt.show()

    def display_result(self,model=None):
        if model==None:
            model=self.model
        img=render_result(self.img,model,self.res)
        plt.imshow(img)
        plt.axis('off')
        plt.show()


    def render_result(self,model):
        render_result(self.img,model,self.res).show()
        
    def render_image(self):
        display_image(self.img)

 
class PatchMatrix:
    def __init__(self,model,image: Union[str,np.ndarray], patch_size: Tuple[int, int], stride: Tuple[Union[int, float], Union[int, float]], approach: str = "dynamic",color_map="BGR"):
   
      
        patches=list(patchify_image(image,patch_size,stride,approach,color_map))
        images=[p[2] for p in patches]
       
        yolo_res=model(images)

        
        
  
        self.patches = patches
        self.yolo_res = yolo_res
        self.stride_x=stride[0]
        self.stride_y=stride[1]
        self.class_map=yolo_res[0].names

        # Determine the maximum x and y to define matrix dimensions
        max_x = 0
        max_y = 0
        for i, x in enumerate(patches):
            if x[0] == 0 and i != 0:
                max_y += 1
            if max_y == 0:
                max_x += 1
        max_y = max_y + 1
        # Initialize a 2D matrix with None
        self.patch_matrix = [[None for _ in range(max_x)] for _ in range(max_y)]
        print(max_x, max_y)
        i=0
        for y in range(max_y):
            for x in range(max_x):
                pd = PatchDetection(x, y, yolo_res[i], patches[i][2],model)
                self.patch_matrix[y][x] = pd
                i+=1


    def merge_patches_and_boxes(self, iou_threshold=0.5, combine_approach="nms"):
        patch_height, patch_width = self.patch_matrix[0][0].img.shape[:2]
        num_rows = len(self.patch_matrix)
        num_cols = len(self.patch_matrix[0])

        # Calculate full image size without considering black borders
        full_height = (num_rows - 1) * self.stride_y + patch_height
        full_width = (num_cols - 1) * self.stride_x + patch_width

        # Create an empty canvas for the full image
        full_image = Image.new('RGB', (full_width, full_height))

        all_boxes = []

        # Iterate through patches and stitch them together
        for row in range(num_rows):
            for col in range(num_cols):
                patch = self.patch_matrix[row][col]
                img = cv2.cvtColor(patch.img, cv2.COLOR_BGR2RGB)

                # Calculate the offsets
                x_offset = col * self.stride_x
                y_offset = row * self.stride_y

                # Paste the image patch on the full image canvas
                full_image.paste(Image.fromarray(img), (x_offset, y_offset))

                # Adjust bounding boxes
                for box in patch.boxes:
                    x_min, y_min, x_max, y_max = tuple([float(f) for f in box.xyxy[0].cpu()])

                    # Ignore black borders: Ensure coordinates are within the image dimensions
                    if x_min >= img.shape[1] or y_min >= img.shape[0] or x_max <= 0 or y_max <= 0:
                        continue

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img.shape[1], x_max)
                    y_max = min(img.shape[0], y_max)

                    # Adjust coordinates by offsets
                    adjusted_box = [
                        x_min + x_offset, y_min + y_offset,
                        x_max + x_offset, y_max + y_offset,
                    ]
                    all_boxes.append((adjusted_box, int(box.cls.cpu()), self.class_map[int(box.cls.cpu())], float(box.conf.cpu())))

        # Apply NMS or other combination approaches
        if iou_threshold >= 0:
            if combine_approach == "nms":
                all_boxes = self.nms(all_boxes, iou_threshold)
            elif combine_approach == "merge":
                all_boxes = self.merge_boxes(all_boxes, iou_threshold)
            else:
                all_boxes = self.merge_boxes(all_boxes, iou_threshold)
                all_boxes = self.nms(all_boxes, iou_threshold)

        return full_image, all_boxes,full_height,full_width,x_offset,y_offset
    
    def nms(self,boxes,iou_threshold=0.5):
        boxes_cord = np.array([box[0] for box in boxes])
        scores = np.array([box[3] for box in boxes])
        boxes_tensor = torch.tensor(boxes_cord, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        indices = nms(boxes_tensor, scores_tensor, iou_threshold)

        # Filter boxes and labels
        filtered_boxes = [boxes[i] for i in indices]

        return filtered_boxes
    
    def merge_boxes(self, boxes, iou_threshold=0.5):
        def compute_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            iou = inter_area / (box1_area + box2_area - inter_area)
            return iou

        merged_boxes = []
        while boxes:
            base_box = boxes.pop(0)
            boxes_to_merge = [base_box]
            boxes_to_check = []
            for box in boxes:
                if base_box[1] == box[1]:  # Ensure same class
                    iou = compute_iou(base_box[0], box[0])
                    if iou > iou_threshold:
                        boxes_to_merge.append(box)
                    else:
                        boxes_to_check.append(box)
                else:
                    boxes_to_check.append(box)
            merged_box = self.merge_box_list(boxes_to_merge)
            merged_boxes.append(merged_box)
            boxes = boxes_to_check

        return merged_boxes

    def merge_box_list(self, boxes):
        x_min = min([box[0][0] for box in boxes])
        y_min = min([box[0][1] for box in boxes])
        x_max = max([box[0][2] for box in boxes])
        y_max = max([box[0][3] for box in boxes])
        class_id = boxes[0][1]
        label = boxes[0][2]
        confidence = max([box[3] for box in boxes])
        return ([x_min, y_min, x_max, y_max], class_id, label, confidence)




    def draw_bounding_boxes(self,figsize=(20, 20), label_font_size=8, box_line_width=1,combine_approach="nms",iou_threshold=0.5):
        """
        Draws bounding boxes on the given image with enhanced visualization.
        
        Parameters:
        - image: PIL Image or numpy array representing the image.
        - boxes: List of bounding boxes, where each bounding box is a tuple of the form:
                ([xmin, ymin, xmax, ymax], class_id, label)
        - figsize: Tuple specifying the figure size (width, height) in inches.
        - label_font_size: Integer specifying the font size of the labels.
        - box_line_width: Integer specifying the line width of the bounding boxes.
        """

        image, boxes=self.merge_patches_and_boxes(iou_threshold,combine_approach)
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)

        for box in boxes:
            (x_min, y_min, x_max, y_max), class_id, label,conf = box
            rect = mpatches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=box_line_width, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Optionally, add label text
            ax.text(x_min, y_min - 5, f'{label}: {conf:.2f}', color='red', fontsize=label_font_size, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.axis('off')
        plt.show()


    def __getitem__(self, idx):
        y,x=idx
        return self.patch_matrix[y][x]

    def __str__(self):
        matrix_str = f"PatchMatrix(rows={len(self.patch_matrix)}, cols={len(self.patch_matrix[0]) if self.patch_matrix else 0})\n"
        for row in self.patch_matrix:
            for pd in row:
                matrix_str += str(pd) + "\n"
        return matrix_str


def display_image(image_source,figsize=(15,15)):
    """
    Displays an image given a file path, image byte stream, or a NumPy array (tensor).
    
    Parameters:
    image_source (str, bytes, or np.ndarray): The path to the image file, the image byte stream, or a NumPy array (tensor).
    """
    if isinstance(image_source, str):
        # Read image from file path
        img = cv2.imread(image_source)
    elif isinstance(image_source, bytes):
        # Read image from byte stream
        nparr = np.frombuffer(image_source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_source, np.ndarray):
        # Use the image array directly
        img = image_source
    else:
        print("Error: Invalid image source type. Provide a file path, image byte stream, or NumPy array.")
        return

    # Check if the image was successfully read
    if img is not None:
        # Convert the image from BGR to RGB if it's not already in RGB format
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Display the image using matplotlib
        plt.figure(figsize=figsize)
        plt.imshow(img_rgb)
        plt.axis('off')  # Hide the axis
        plt.show()
    else:
        print("Error: Could not load image.")




def patchify_image(image: Union[str,np.ndarray], patch_size: Tuple[int, int], stride: Tuple[Union[int, float], Union[int, float]], approach: str = "dynamic",color_map="BGR") -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Splits an image into patches using a sliding window approach with different methods to handle edge cases.

    Parameters:
    image (np.ndarray): The input image to be split into patches.
    patch_size Tuple[int, int]: The size of the window (width, height).
    stride (Tuple[Union[int, float], Union[int, float]]): The stride (step size) to move the window. If float, it is treated as a percentage of window size. If int, it is treated as the number of pixels.
    approach (str): Method to handle edge cases. Options are 'padding', 'dynamic', or 'tune'.
    verbose (bool): If True, prints additional information about the process.

    Yields:
    Tuple: A tuple containing the x and y coordinates of the top-left corner of the window and the window image.
    """

    if isinstance(image,str):
        image=cv2.imread(image)
        if color_map=="RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
    img_height, img_width = image.shape[:2]


    if not isinstance(stride, tuple):
        raise ValueError("stride must be a tuple of int or float.")

    if isinstance(stride[0], float):
        stride_x = int(patch_size[0] * stride[0])
    else:
        stride_x = stride[0]

    if isinstance(stride[1], float):
        stride_y = int(patch_size[1] * stride[1])
    else:
        stride_y = stride[1]

    stride = (stride_x, stride_y)

    if approach == "padding":
        # Calculate padding required to make the image dimensions divisible by the window size
        pad_x = (patch_size[0] - (img_width % patch_size[0])) % patch_size[0]
        pad_y = (patch_size[1] - (img_height % patch_size[1])) % patch_size[1]

        # Pad the image with a constant value (black)
        padded_image = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Slide window over the padded image
        for y in range(0, padded_image.shape[0] - patch_size[1] + 1, stride[1]):
            for x in range(0, padded_image.shape[1] - patch_size[0] + 1, stride[0]):
                window = padded_image[y:y + patch_size[1], x:x + patch_size[0]]
                yield (x, y, window)

    elif approach == "dynamic":
        # Slide window over the image, adjusting window size if it goes out of bounds
        prev_x, prev_y = -1, -1
        for y in range(0, image.shape[0], stride[1]):
            for x in range(0, image.shape[1], stride[0]):
                x_end = min(x + patch_size[0], image.shape[1])
                y_end = min(y + patch_size[1], image.shape[0])
                x_start = x
                y_start = y

                # Extract window
                window = image[y_start:y_end, x_start:x_end]

                # Skip if the patch is fully inside the previous patch
                if prev_x != -1 and prev_y != -1:
                    if x_start >= prev_x and y_start >= prev_y and x_end <= prev_x + patch_size[0] and y_end <= prev_y + patch_size[1]:
                        continue

                # Update previous patch coordinates
                prev_x, prev_y = x_start, y_start

                yield (x_start, y_start, window)
    else:
        raise ValueError("Invalid approach. Must be one of 'padding', 'dynamic'.")
    




  ###################################
  # def process_image_with_sliding_window(image, model, step_size, patch_size):
#     height, width = image.shape[:2]
#     results = []
#     scores = []

#     for (x, y, window) in patchify_image(image, step_size, patch_size):
#         window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
#         window_pil = Image.fromarray(window_rgb)
#         result = model(window_pil)
        
#         for bbox, score in zip(result[0].boxes.xyxy.clone(), result[0].boxes.conf.clone()):
#             bbox[0] += x  # Adjust the x-coordinate of the top-left corner
#             bbox[1] += y  # Adjust the y-coordinate of the top-left corner
#             bbox[2] += x  # Adjust the x-coordinate of the bottom-right corner
#             bbox[3] += y  # Adjust the y-coordinate of the bottom-right corner
#             results.append(bbox.numpy())
#             scores.append(score.item())
    
#     return np.array(results).astype(np.float32), np.array(scores).astype(np.float32), height, width

# def apply_nms(boxes, scores, iou_threshold=0.5):
#     boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
#     scores_tensor = torch.tensor(scores, dtype=torch.float32)
#     indices = nms(boxes_tensor, scores_tensor, iou_threshold)
#     return indices

# def merge_nearby_boxes(boxes, scores, proximity_threshold, iou_threshold=0.5):
#     indices = apply_nms(boxes, scores, iou_threshold)
#     filtered_boxes = boxes[indices]
#     filtered_scores = scores[indices]

#     merged_boxes = []
#     merged_scores = []

#     while len(filtered_boxes) > 0:
#         max_score_idx = np.argmax(filtered_scores)
#         max_score_box = filtered_boxes[max_score_idx]
#         max_score = filtered_scores[max_score_idx]

#         ious = compute_iou(max_score_box, filtered_boxes)
#         proximities = compute_proximity(max_score_box, filtered_boxes)
#         overlapping_indices = np.where((ious > iou_threshold) | (proximities < proximity_threshold))[0]

#         if len(overlapping_indices) > 1:
#             merged_box = np.mean(filtered_boxes[overlapping_indices], axis=0)
#             merged_score = np.mean(filtered_scores[overlapping_indices])
#         else:
#             merged_box = max_score_box
#             merged_score = max_score

#         merged_boxes.append(merged_box)
#         merged_scores.append(merged_score)

#         filtered_boxes = np.delete(filtered_boxes, overlapping_indices, axis=0)
#         filtered_scores = np.delete(filtered_scores, overlapping_indices)

#     final_boxes, final_scores = remove_inside_boxes(np.array(merged_boxes), np.array(merged_scores))

#     return final_boxes, final_scores

# def compute_iou(box, boxes):
#     x_min = np.maximum(box[0], boxes[:, 0])
#     y_min = np.maximum(box[1], boxes[:, 1])
#     x_max = np.minimum(box[2], boxes[:, 2])
#     y_max = np.minimum(box[3], boxes[:, 3])

#     intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
#     box_area = (box[2] - box[0]) * (box[3] - box[1])
#     boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

#     union = box_area + boxes_area - intersection
#     iou = intersection / union
#     return iou

# def compute_proximity(box, boxes):
#     """
#     Compute proximity between a box and an array of boxes.
#     :param box: Single bounding box [x_min, y_min, x_max, y_max].
#     :param boxes: Array of bounding boxes [[x_min, y_min, x_max, y_max], ...].
#     :return: Array of proximity values.
#     """
#     x_min_dist = np.abs(box[0] - boxes[:, 0])
#     y_min_dist = np.abs(box[1] - boxes[:, 1])
#     x_max_dist = np.abs(box[2] - boxes[:, 2])
#     y_max_dist = np.abs(box[3] - boxes[:, 3])

#     proximity = np.minimum(x_min_dist, x_max_dist) + np.minimum(y_min_dist, y_max_dist)
#     return proximity

# def remove_inside_boxes(boxes, scores, confidence_threshold=0.9):
#     keep_boxes = []
#     keep_scores = []

#     for i, box in enumerate(boxes):
#         inside = False
#         for j, other_box in enumerate(boxes):
#             if i != j and is_inside(box, other_box):
#                 if scores[i] <= confidence_threshold:
#                     inside = True
#                     break
#         if not inside:
#             keep_boxes.append(box)
#             keep_scores.append(scores[i])

#     return np.array(keep_boxes), np.array(keep_scores)

# def is_inside(box1, box2):
#     """
#     Check if box1 is fully inside box2.
#     :param box1: Single bounding box [x_min, y_min, x_max, y_max].
#     :param box2: Single bounding box [x_min, y_min, x_max, y_max].
#     :return: True if box1 is inside box2, otherwise False.
#     """
#     return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]

# def render_combined_results(image, results, scores, model, step_size, proximity_threshold, iou_threshold=0.5):
#     combined_image = image.copy()
#     merged_boxes, merged_scores = merge_nearby_boxes(results, scores, proximity_threshold, iou_threshold)

#     for box, score in zip(merged_boxes, merged_scores):
#         x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#         cv2.rectangle(combined_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
#         cv2.putText(combined_image, f"Detected: {score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     return combined_image





    
