#!/usr/bin/env python -W ignore::DeprecationWarning

import argparse
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import pdb
import os
import time
from tqdm import tqdm

#=====================================
#Metadata
#=====================================
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval().cuda()

im2tensor_nonorm = T.Compose([
    T.ToTensor()
])

#=====================================
#inference
#=====================================
def transform_nonorm(images):
    if not isinstance(images, torch.Tensor):
        images = torch.cat(
            [im2tensor_nonorm(i)[None, :, :, :].cuda() for i in images], dim=0)
    return images

def infer_keypoint_detection(images):
    with torch.no_grad():
        x = transform_nonorm(images)
        output = model(x)
        return output[0]

# visualization code
def visualize_prediction(img_path, pred, threshold=0.75, rect_th=3, text_size=3, text_th=3):
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(pred_boxes)):
        cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img, pred_class[i], pred_boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th) # Write the prediction class
    pic = plt.figure(figsize=(20,30)) # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    pic.savefig('temp.png')

#=====================================
#filtration
#=====================================
def filter_confidence(fname, pred, threshold):
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'].cpu().numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].cpu().detach().numpy())] # Bounding boxes
    pred_score = list(pred['scores'].cpu().detach().numpy())

    # Get list of index with score greater than threshold.
    list_large_score = [pred_score.index(x) for x in pred_score if x > threshold]
    #import pdb;pdb.set_trace()
    if not list_large_score:
        return [], [], []
    else: 
        pred_t = list_large_score[-1] 
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
    return pred_boxes, pred_score, pred_class

def filter(bboxes, scores, classes):

    def filter_number_of_humans(bboxes, scores, classes):
        label_human = [i+1 for i in range(len(classes)) if classes[i] == 'person']
        if len(label_human) != 1:
            return False
        else:
            return True

    def filter_scale_of_human(bboxes, scores, classes, 
                              scale_upper_bound=args.upper_bound, 
                              scale_lower_bound=args.lower_bound):
        human_box = bboxes[classes.index('person')]
        human_box_area = abs(human_box[0][0]-human_box[1][0])*abs(human_box[0][1]-human_box[1][1])
        full_area = float(image_array.shape[0])*float(image_array.shape[1])
        ratio = human_box_area / full_area
        #pdb.set_trace()
        #if ratio <= scale_upper_bound and ratio >= scale_lower_bound:
        ratio = float(ratio)
        if ratio <= float(scale_upper_bound) and ratio >= float(scale_lower_bound):
            return True
        else:
            return False

    def filter_position_of_human(bboxes, scores, classes, margin_threshold=50):
        human_box = bboxes[classes.index('person')]
        image_height = image_array.shape[0]
        image_width = image_array.shape[1]
        # check left and upper boundary
        if human_box[0][0] < margin_threshold or human_box[0][1] < margin_threshold:
            return False
        # check right and lower boundary
        if image_width - human_box[1][0] < margin_threshold or image_height - human_box[1][1] < margin_threshold:
            return False
        return True

    # discard if there's no human or more than one human
    f1 = filter_number_of_humans(bboxes, scores, classes)
    if not f1:
        return False, "1"
    
    # discard if human bounding box is too large or too small
    f2 = filter_scale_of_human(bboxes, scores, classes)
    if not f2:
        return False, "2"

    # discard if human bounding box is too close to the boundary
    f3 = filter_position_of_human(bboxes, scores, classes)
    if not f3:
        return False, "3"

    return True, "0"

#===================================== Hajime!
parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--video_folder_path', required=True)
parser.add_argument('--result_folder_path', required=True)
parser.add_argument('--stats_file_path', required=True)
parser.add_argument('--upper_bound', required=True)
parser.add_argument('--lower_bound', required=True)
parser.add_argument('--confidence_threshold', required=True)
args = parser.parse_args()

if not os.path.isdir(args.result_folder_path):
    os.mkdir(args.result_folder_path)

video_path = os.path.join(args.video_folder_path, args.video)
result_path = os.path.join(args.result_folder_path, f"{args.video}_filtered_conf_{args.confidence_threshold}_ub_{args.upper_bound}_lb_{args.lower_bound}") 
if not os.path.isdir(result_path):
    os.mkdir(result_path)
img_names = os.listdir(video_path)
num_frames = len(img_names)

# process each frame in the video
checkpoint1 = time.time()
print(f"Processing {args.video}......")
print(f"UB: {args.upper_bound}, LB: {args.lower_bound}")
counter,keep,discard = 0,0,0
e1,e2,e3 = 0,0,0

for frame_idx in tqdm(range(len(img_names))): 
    frame = img_names[frame_idx]
    counter = counter + 1
    img_path = os.path.join(video_path, frame)
    #filtered_img_path = os.path.join(result_path, frame)
    
    # inference
    image = Image.open(img_path)
    image = image.convert('RGB')
    image_array = np.array(image)
    obj_predictions = infer_keypoint_detection([image_array])

    # we only keep bboxes with confidence score higher than the threshold
    pred_boxes, pred_score, pred_class = filter_confidence(frame, obj_predictions, threshold=float(args.confidence_threshold))

    # filtering based on standards above
    filter_flag, error_type = filter(pred_boxes, pred_score, pred_class)

    #print(f'Frame: {counter}/{num_frames}: {filter_flag}')

    # keep or discard
    if 'True' in str(filter_flag):
        new_frame_name = str(keep).zfill(10) + ".png"
        filtered_img_path = os.path.join(result_path, new_frame_name)
        #filtered_img_path = os.path.join(result_path, frame)
        copy_img = subprocess.run(['cp', str(img_path), str(filtered_img_path)],
                                    stdout=subprocess.PIPE)
        keep += 1
    elif 'False' in str(filter_flag):
        if error_type == "1": 
            e1 = e1 + 1
        elif error_type == "2": 
            e2 = e2 + 1
        else:
            e3 = e3 + 1
        discard += 1
    #if counter % 500 == 0:   
    #    print(f'{counter}/{num_frames}; keep: {keep}; discard: {discard}')

# write statistics to stat file
with open(args.stats_file_path, 'a+') as f:
    f.write(f"Video name: {args.video}, Usable frames: {keep}/{num_frames}\n")
    f.write(f"Error 1: {e1}/{num_frames-keep}, Error 2: {e2}/{num_frames-keep}, Error 3: {e3}/{num_frames-keep}\n")
checkpoint2 = time.time()
time_overhead = checkpoint2 - checkpoint1
print(f"Time used is: {time_overhead}")