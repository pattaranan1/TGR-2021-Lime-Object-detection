import cv2
import numpy as np
import tensorflow as tf
import os
from step_2_2_problem import *
import time

if __name__ == '__main__':

    cap = cv2.VideoCapture('data/clips/test_clip.h264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    sqsize = max(width,height)
    
    LINE1 = int(sqsize/2-100) + 150
    LINE2 = int(sqsize/2+100) + 200
    
    lime_count = 0
    marker_count = 0
    entry = False
    
    total_marker_width = 0
    MARKER_DIAMETER = 0.04
    pixel_per_metric = 0
    lime_sizes = []

    skip = 0

    start_time = time.time()
    while True:
        # capture image
        ret,raw_img = cap.read()

        if not ret:
            break
        # add margin
        frame = np.zeros((sqsize,sqsize,3), np.uint8)
        if width > height:
            offset = int( (width - height)/2 )
            frame[offset:height+offset,:] = raw_img
        else:
            offset = int( (height - width)/2 )
            frame[:,offset:] = raw_img

        # problems
        # detect every 3 frame
        if skip == 0:
            skip = 2

            bboxes, classes, scores = detect_objects(frame)

            # counting objects and measure diameter of lime
            if np.squeeze(bboxes[scores > 0.8].size > 0):
                box = np.squeeze(bboxes)[0]
                ymin = box[0] * height
                xmin = box[1] * width
                ymax = box[2] * height
                xmax = box[3] * width

                if xmin < LINE2 and xmax > LINE1 and not entry:
                    entry = True
                
                if entry and xmax <= LINE1:
                    entry = False

                    if(classes[scores > 0.8][0] == 1):
                        lime_count += 1
                        lime_diameter = ((xmax - xmin) + (ymax - ymin))  / (2 * pixel_per_metric)
                        lime_sizes.append(lime_diameter)
                        print(f'lime {lime_count} diameter: {lime_diameter * 1000:.3f} mm')

                    elif(classes[scores > 0.8][0] == 2):
                        marker_count += 1
                        total_marker_width += ((xmax - xmin) + (ymax - ymin)) / 2
                     
                        pixel_per_metric = (total_marker_width / marker_count) / MARKER_DIAMETER

            label_id_offset = 1
            
        else:
            skip = skip - 1

        img = overlay_objects(frame, bboxes, classes, scores)

        # insert Lime Count information text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            'Lime Count: ' + str(lime_count),
            (10, 35),
            font,
            0.8,
            (0, 0xFF, 0xFF),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
            )
        
        # insert Marker Count information text
        cv2.putText(
            img,
            'Marker Count: ' + str(marker_count),
            (10, 55),
            font,
            0.8,
            (0, 0xFF, 0xFF),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
            )

        # overlay with line
        pt1 = ( LINE1, 0 ) 
        pt2 = ( LINE1, int(sqsize) ) 
        
        cv2.line(img, pt1, pt2, (0,0,255), 2)
        pt1 = ( LINE2, 0 ) 
        pt2 = ( LINE2, int(sqsize) ) 
        cv2.line(img, pt1, pt2, (0,0,255), 2) 

        # preview image
        cv2.imshow('Preview', img)     
        key = cv2.waitKey(int(1000/fps))
        if key == ord('q'):
            break

    cap.release()
    elapsed_time = time.time() - start_time
    print_result(lime_count, marker_count, lime_sizes, elapsed_time)