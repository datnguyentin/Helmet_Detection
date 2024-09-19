import utils
import model_function as mf
from collections import defaultdict
import cv2
import numpy as np
import time 
import streamlit as st
from cap_from_youtube import cap_from_youtube
import os
import csv

#Initialize models

def process(video, run_option = "Continuous", app_running = False, frame_placeholder = None):

    waitKey = 1 if run_option == "Continuous" else 0

    helmet_detector, helmet_label_map = mf.load_helmet_detector()
    plate_ocr, char_label_map = mf.load_plate_ocr()
    tracker = mf.load_tracker()


    #Initialize tracking_record

    motor_record = defaultdict(lambda: utils.motor_tracking())
    plate_record = defaultdict(lambda: utils.plate_tracking())

    #Check source from youtube or video/webcam
    if video.startswith("http"):
        cap = cap_from_youtube(video, 'best')
    else:
        cap = cv2.VideoCapture(video)
    
    if cap is None or not cap.isOpened():
        st.text("No video source found!")
        return False

    if not os.path.isfile('out/Violation_Record.csv'):

        fieldnames = ['Date', "Time", "Plate", "Path_to_image"]

        with open("out/Violation_Record.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames )
            writer.writeheader()

    frame_count = 0
    prev_frame_time = 0
    start_time = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        frame_copy = frame.copy()
        

        if frame_count % 2 != 0:
            utils.visualize_image(frame, helmet_label_map, helmet_visualize_l, coord2id, plate_visualize_l)
            frame_count += 1

            for coord in multiplate_check.keys():
                id = coord2id[coord]
                plate_ = plate_record[id]
                plate_['next_detect'] -= 1

                if plate_['next_detect'] == 0:

                    plate_image = plate_['pending_image']
                    plate_text = mf.detect_plate(plate_image, plate_ocr, char_label_map)
        
                    if plate_text is not None:
                        plate_[plate_text] += 1

                    plate_['next_detect'] = mf.get_next_detect(plate_, plate_text)
                    plate_['pending_image'] = None   

            
            time.sleep(0.001)

        
        else:
            frame_count += 1

            #Initialized necessary list for later use
            coord2id = {}
            non_helmet_l = []
            motor_l = []
            plate_l = []

            helmet_visualize_l = []
            plate_visualize_l = []

            helmet_results = helmet_detector.predict(frame, verbose = False)[0].cpu()
            boxes = np.array(helmet_results.boxes.data[:, :4])
            scores = np.array(helmet_results.boxes.data[:, 4])
            filter_resulted = mf.non_max_suppression(boxes, scores, 0.5)
            helmet_results = helmet_results[filter_resulted]

            #Classify each object to their list
            for item in helmet_results.boxes.data:
                x, y, X, Y, cf, cls = item
                x, y, X, Y, cf, cls = int(x), int(y), int(X), int(Y), float(cf), int(cls)
                if cls == 0:
                    motor_l.append([x, y, X, Y, cf, cls])
                elif cls == 2 or cls == 3:
                    helmet_visualize_l.append([x, y, X, Y, cf, cls])
                    if cls == 3:
                        non_helmet_l.append([x, y, X, Y, cf])
                elif cls == 1:
                    plate_l.append([x, y, X, Y])


            #Update Tracker
            if len(motor_l) != 0:
                tracking_results = tracker.update(np.array(motor_l), frame)
                for tracking_obj in tracking_results:
                    coord2id = utils.assign_id_to_coord(tracking_obj, motor_l, coord2id)

            else:
                tracking_results = tracker.update(np.empty((0, 6)), frame)

            #Ensure every new bike id in motor_tracking list
            for bike_id in coord2id.values():
                flag = motor_record[bike_id]

            #Update Motor Record
            motor_record, id_results = utils.update_motor_tracking(motor_record, coord2id)

            #Save the non-motorcyclist object
            if len(id_results) != 0:
                for id_re in id_results:
                    bike_id, violated = id_re
                    if violated:
                        vi_image, vi_plate = utils.save_results(motor_record[bike_id], plate_record[bike_id])
                        if app_running and vi_image is not None:
                            st.sidebar.image(vi_image, vi_plate)
                    del(motor_record[bike_id])
                    if bike_id in plate_record.keys():
                        del(plate_record[bike_id])

            
            multiplate_check = dict()                                                                                                            
            for plate in plate_l:
                xb, yb, Xb, Yb = utils.get_bike_related(plate, list(coord2id.keys()))
        
                if xb != None:
                
                    bike_id = coord2id[(xb, yb, Xb, Yb)]
                    xp, yp, Xp, Yp = plate
                    plate_ = plate_record[bike_id]
        
                    if (xb, yb, Xb, Yb) not in multiplate_check.keys():
                        plate_['next_detect'] -= 1
                        multiplate_check[(xb, yb, Xb, Yb)] = [(xp, yp, Xp, Yp), None]
                    else:
                        true_plate = utils.compare_plate(plate, multiplate_check[(xb, yb, Xb, Yb)][0], (xb, yb, Xb, yb))
                        if true_plate != plate:
                            continue
                        else:
                            old_plate = multiplate_check[(xb, yb, Xb, Yb)][1]
                            if old_plate != None:
                                plate_[old_plate] -= 1
                                                   
                    display_plate = mf.get_final_plate(plate_)  
                    plate_visualize_l.append([xp, yp, Xp, Yp, display_plate])
                    
                    if plate_['next_detect'] == 0:
                        plate_img = frame[yp:Yp, xp:Xp]
                        plate_text = mf.detect_plate(plate_img, plate_ocr, char_label_map)
        
                        if plate_text is not None:
                            plate_[plate_text] += 1
                            multiplate_check[(xb, yb, Xb, Yb)][1] = plate_text
                        plate_['next_detect'] = mf.get_next_detect(plate_, plate_text)
                    
                    elif plate_['next_detect'] == 1:
                        plate_img = frame[yp:Yp, xp:Xp]
                        plate_['pending_image'] = plate_img


            #Create record for helmet
            for non_helmet in non_helmet_l:
                xb, yb, Xb, Yb = utils.get_bike_related(non_helmet, list(coord2id.keys()))
                if xb is not None:
                    xh, yh, Xh, Yh, conf = non_helmet
                    bike_id = coord2id[(xb, yb, Xb, Yb)]
                    nhelmet_ = motor_record[bike_id]
                    nhelmet_['non_helmet'] += 1
                    if  conf > nhelmet_['confidence'] and nhelmet_['total_detections'] >= 16:
                        nhelmet_['confidence'] = conf
                        nhelmet_['violation_image'] = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)


        for key in plate_record.keys():
            display = plate_record[key]
            print("ID:", key)
            for k, v in display.items():
                if k != "pending_image":
                    print('\t' + k + ": ", v)                                                  #delete when done
        print('--------------------------------------------')

        utils.visualize_image(frame, helmet_label_map, helmet_visualize_l, coord2id, plate_visualize_l)


        fps = 1 / (start_time - prev_frame_time)
        prev_frame_time = start_time
        utils.draw_fps(fps, frame)

        if frame_placeholder == None:
            resized_frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow('Helmet Detection', resized_frame)   
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame) 
        # Wait for key press
        key = cv2.waitKey(waitKey) & 0xFF
        
        # If 'n' is pressed, continue to next frame
        if key == ord('n'):
            continue
        # If 'q' is pressed, quit
        if key == ord('p'):
            cv2.waitKey(-1)

        if key == ord('q'):
            break
    

    cap.release()
    cv2.destroyAllWindows() 