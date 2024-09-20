import re
import numpy as np
from boxmot import BYTETracker
from ultralytics import YOLO
import cv2
import os
from collections import defaultdict
import model_function as mf
import datetime
import csv
import math

##################################################
#                  TRACKING LISTS                #
##################################################
class plate_tracking(defaultdict):
    def __init__(self):
        super().__init__(lambda: 0)      #Store the occurence of each plate value
        self['next_detect'] = 1          #The number of remaining frame until the next detection
        self['pending_image'] = None

class motor_tracking(dict):
    def __init__(self):
        super().__init__()
        self['total_detections'] = 0      #Total number of frames detected
        self['non_helmet'] = 0            #Total non_helmet objects detected in the motorcyclist objects in every frames
        self['violation_image'] = None    # Containing violtation image   
        self['confidence'] = 0            #Save highest confidence score for violation image     
        self['n_frames_none_detect'] = 0  #number of consecutive frames objects didn't show up (set to zero if it did)
    
##################################################
#               UPDATE TRACKING LISTS            #
##################################################


def assign_id_to_coord(tracking_obj, motor_l, coord2id):
    score = round(tracking_obj[5], 5)
    x, y, X, Y = 0, 0, 0, 0
    for obj in motor_l:
        x_t, y_t, X_t, Y_t, cf, _ = obj
        if round(cf, 5) == score:
            x, y, X, Y = round(x_t), round(y_t), round(X_t), round(Y_t)  
    id = tracking_obj[4]
    coord2id[(x, y, X, Y)] = id

    return coord2id

def update_motor_tracking(motor_record, coord2id):
    id_results = []
    for bike_id in list(motor_record.keys()):
        if bike_id in coord2id.values():
            motor_record[bike_id]['total_detections'] += 1
            motor_record[bike_id]['n_frames_none_detect'] = 0
        else:
            motor_record[bike_id]['n_frames_none_detect'] += 1

            if motor_record[bike_id]['n_frames_none_detect'] == 40:
                non_helmet_frames = motor_record[bike_id]['non_helmet']
                total_frames_detected = motor_record[bike_id]['total_detections']

                non_helmet = (non_helmet_frames / total_frames_detected) >= 0.5
                id_results.append([int(bike_id), non_helmet])
    
    return motor_record, id_results

def save_results(motor_record, plate_record, out_folder = 'out/images'):
    os.makedirs(out_folder, exist_ok = True)
    most_common_license = mf.get_final_plate(plate_record)

    image = motor_record['violation_image']

    #Save images and license plate
    if image is not None:
        current_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        path_to_image = f'{out_folder}/{most_common_license}_{current_time}.jpg'
        saved_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_to_image, saved_image)

        #Write csv file
        with open("out/Violation_Record.csv", 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            today = datetime.date.today().strftime("%d-%m-%y")
            timenow = datetime.datetime.today().strftime("%H:%M:%S")
            writer.writerow([today, timenow, most_common_license, path_to_image])


        #Return for display streamlit
        return np.array(image), most_common_license
    return None, None

    #Return for displaying on streamlit

##################################################
#          GET OBJECT RELATING TO MOTOR          #
##################################################


def get_bike_related(obj, vehicles):
    x1, y1, x2, y2 = obj[:4]

    for vehicle in vehicles:
        x_b, y_b, X_b, Y_b = vehicle[:4]

        if x1 > x_b - 20 and y1 > y_b - 20 and x2 < X_b + 20 and y2 < Y_b + 20:
            return x_b, y_b, X_b, Y_b

    return None, None, None, None

def center_of_box(coord):

        xmin, ymin, xmax, ymax = coord
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        return center_x, center_y

def calculate_angle(coord_1, coord_2):
    '''Calculate the angle create by two coord 1 and coord 2 with the vertical line through coord_1'''
    x1, y1 = coord_1
    x2, y2 = coord_2

    delta_x = x2 - x1
    delta_y = y2 - y1

    angle = 180 - abs(math.degrees(math.atan2(delta_x, delta_y)))

    return angle



def compare_plate(coord_plate_1, coord_plate_2, coord_bike):

    x_center_p1, y_center_p1 = center_of_box(coord_plate_1)
    x_center_p2, y_center_p2 = center_of_box(coord_plate_2)
    x_center_b, y_center_b = center_of_box(coord_bike)

    angle_plate1 = calculate_angle((x_center_b, y_center_b), (x_center_p1, y_center_p1))
    angle_plate2 = calculate_angle((x_center_b, y_center_b), (x_center_p2, y_center_p2))

    if angle_plate1 < angle_plate2:
        return coord_plate_1  # Plate 1 is closer to a vertical line with the bike
    else:
        return coord_plate_2
    
def check_helmet_position(coord_helmet, coord_bike):
    x_center_h, y_center_h = center_of_box(coord_helmet)
    x_center_b, y_center_b = center_of_box(coord_bike)
    print(y_center_h, y_center_b)
    if y_center_h < y_center_b:
        return True
    return False



##################################################
#                   VISUALIZE                    #
##################################################
def visualize_image(img, label_map, helmet_results = None, coord2id = None, plate_result = None):
    if helmet_results:
        color_map = {2:  [48, 185, 90], 3: [47, 47, 231]}
        for re in helmet_results:
            x, y, X, Y, cf, cls = re
            x, y, X, Y, cls = int(x), int(y), int(X), int(Y), int(cls)
            box_color = color_map[cls]
            text_color = (255, 255, 255)

            labels = label_map[cls]

            # Draw object box
            first_half_box = (x, y)
            second_half_box = (X, Y)
            cv2.rectangle(img, first_half_box, second_half_box, box_color, 2)

            text_print = '{label} {con:.2f}'.format(label = labels, con = cf)
            text_location = (x, y - 10 )
            labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)

            # Draw text's background
            cv2.rectangle(img, 
                          (x, y - labelSize[1] - 10),
                          (x + labelSize[0], y + baseLine-10)
                          ,box_color , cv2.FILLED)
            
            cv2.putText(img, text_print ,text_location, 
                        cv2.FONT_HERSHEY_SIMPLEX , 1, 
                        text_color, 1, cv2.LINE_AA)

        if coord2id:
            for k, v in coord2id.items():
                id = v
                x, y, X, Y = k

                box_color = [206, 126, 0]
                text_color = (255, 255, 255)

                labels = f'{int(id)}'

                first_half_box = (x, y)
                second_half_box = (X, Y)
                cv2.rectangle(img, first_half_box, second_half_box, box_color, 2)
                
                text_print = '{label}'.format(label = labels)
                text_location = (x, y - 10 )
                labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    
                # Draw text's background
                cv2.rectangle(img, 
                              (x, y - labelSize[1] - 10),
                              (x + labelSize[0], y + baseLine-10)
                              ,box_color , cv2.FILLED)
                
                cv2.putText(img, text_print ,text_location, 
                            cv2.FONT_HERSHEY_SIMPLEX , 1, 
                            text_color, 1, cv2.LINE_AA)
        if plate_result:
            for re in plate_result:
                x, y, X, Y, text = re
                box_color = [206, 126, 0]
                text_color = (255, 255, 255)

                labels = text

                first_half_box = (x, y)
                second_half_box = (X, Y)
                cv2.rectangle(img, first_half_box, second_half_box, box_color, 2)
                
                text_print = '{label} {con:.2f}'.format(label = labels, con = cf)
                text_location = (x, y - 10 )
                labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    
                # Draw text's background
                cv2.rectangle(img, 
                              (x, y - labelSize[1] - 10),
                              (x + labelSize[0], y + baseLine-10)
                              ,box_color , cv2.FILLED)
                
                cv2.putText(img, text_print ,text_location, 
                            cv2.FONT_HERSHEY_SIMPLEX , 1, 
                            text_color, 1, cv2.LINE_AA)
            

def draw_fps(fps, img):
  fps = int(fps)

  cv2.rectangle(img, (10,2), (280,50), (255,255,255), -1)
  cv2.putText(img, "FPS: "+str(fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), thickness=3)

  return img