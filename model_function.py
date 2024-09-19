from boxmot import BYTETracker
from ultralytics import YOLO
import math
import numpy as np
import re


############################################
#                 LOAD MODEL               #
############################################
def load_helmet_detector():
    model_path = "models/Helmet_Detector/best.pt"
    helmet_model = YOLO(model_path, task = 'detect')

    return helmet_model, helmet_model.names

def load_plate_ocr():
    model_path = "models/Yolo_plate/best.pt"
    plate_model = YOLO(model_path, task = 'detect')

    return plate_model, plate_model.names
    
def load_tracker():
    # Define the tracker object
    tracker = BYTETracker(
        track_thresh=0.45,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=30,
    ) 
    return tracker

############################################
#             READ_LICESE_PLATE            #
############################################

def mapping_character(plate_text, n_numbers):
    char_to_int = {'O': '0',
                    'Z': '2',
                    'A': '4',
                    'G': '6',
                    'S': '5'}
    
    int_to_char = { '2': 'Z',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '0': 'O'}
    
    if n_numbers == 8:
        for ix in [0, 1, 4, 5, 6, 7]:
            if plate_text[ix] in char_to_int.keys():
                plate_text[ix] = char_to_int[plate_text[ix]]
    elif n_numbers == 9:
        for ix in [0, 1, 4, 5, 6, 7, 8]:
            if plate_text[ix] in char_to_int.keys():
                plate_text[ix] = char_to_int[plate_text[ix]]
    
    if plate_text[2] in int_to_char.keys():
        plate_text[2] = int_to_char[plate_text[2]]
        
    return plate_text


def check_license_format(string, n_numbers):   
    if n_numbers == 8: 
        pattern = r'^\d{2}[A-Za-z][\dA-Za-z]-\d{4}$'
        return bool(re.match(pattern, string))
    elif n_numbers == 9:
        pattern = r'^\d{2}[A-Za-z][\dA-Za-z]-\d{5}$'
        return bool(re.match(pattern, string))
    elif n_numbers == 10:
        pattern = r'^\d{2}MD\d-\d{5}$'
        return bool(re.match(pattern, string))




def read_license_plate(results, label_map):
    char_list = []
    for result in results.boxes.data.tolist():
        x,y, X, Y, cf, cls = result
        if cf >= 0.3:
            char_list.append([x, y, label_map[int(cls)]])
    
    
    if len(char_list) < 8 or len(char_list) > 10:
        return None
    
    char_list = sorted(char_list, key = lambda x: x[1])

    plate = []
    license_format = len(char_list)

    if len(char_list) == 8:
        upper_chars = char_list[:4]
        upper_chars = sorted(upper_chars, key = lambda x: x[0])
        upper_chars = [char[2] for char in upper_chars]
        plate.extend(upper_chars)
        
        lower_chars = char_list[4:]
        lower_chars =sorted(lower_chars, key = lambda x: x[0])
        lower_chars = [char[2] for char in lower_chars]
        plate.extend(lower_chars)

        
    
    elif len(char_list) == 9:
        upper_chars = char_list[:4]
        upper_chars = sorted(upper_chars, key = lambda x: x[0])
        upper_chars = [char[2] for char in upper_chars]
        plate.extend(upper_chars)
        
        lower_chars = char_list[4:]
        lower_chars =sorted(lower_chars, key = lambda x: x[0])
        lower_chars = [char[2] for char in lower_chars]
        plate.extend(lower_chars)

    elif len(char_list) == 10:
        upper_chars = char_list[:5]
        upper_chars = sorted(upper_chars, key = lambda x: x[0])
        upper_chars = [char[2] for char in upper_chars]
        plate.extend(upper_chars)

        lower_chars = char_list[5:]
        lower_chars =sorted(lower_chars, key = lambda x: x[0])
        lower_chars = [char[2] for char in lower_chars]
        plate.extend(lower_chars)

    filtered_plate = mapping_character(plate, license_format)
    filtered_plate.insert(4, '-')
    plate_text = "".join(filtered_plate)
        

    if check_license_format(plate_text, license_format):
        return plate_text
    
    return None

def detect_plate(plate_img, plate_ocr, plate_label_map):
    
    plate_predict = plate_ocr(plate_img, verbose = False, imgsz = 320, agnostic_nms = True)[0].cpu()
    #boxes = np.array(plate_predict.boxes.data[:, :4])
    #score = np.array(plate_predict.boxes.data[:, 4])

    #filtered_predictions = non_max_suppression(boxes, score, 0.5)
    #plate_predict = plate_predict[filtered_predictions]

    plate_text =  read_license_plate(plate_predict, plate_label_map)
    
    return plate_text

def get_final_plate(plate_record):
    filtered_license = {k:v for k, v in plate_record.items() if k not in['next_detect', 'pending_image']}
    if not filtered_license:
        return "Unidentify"
    most_common_license = max(filtered_license, key=filtered_license.get)

    return most_common_license

def get_next_detect(record, plate_text):
    if plate_text == None:
        return 3
    else:
        occurence = record[plate_text]
        total = sum(value for key, value in record.items() if key not in['next_detect', 'pending_image'])
        next_detect = math.floor((occurence*10)/total) - math.floor(4 / total)
        
        if next_detect == 1 or next_detect == 0:
            return 6
        return next_detect



####################################################
#               NON-MAX SUPPRESSION                #
####################################################


def non_max_suppression(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # Update the indices
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou



