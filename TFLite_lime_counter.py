# TGR-02-Team Ice Cream Chillin' 
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import time

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.75)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')], num_threads=4)
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, num_threads=4)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

elapsed_time = []

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

start_total_time = time.time()

idx = 0
IMG_PATH = './data/images'

sqsize = max(imW,imH)

LINE1 = int(sqsize/2-100) + 150
LINE2 = int(sqsize/2+100) + 200

lime_count = 0
marker_count = 0
entry = False

total_marker_width = 0
MARKER_DIAMETER = 0.04
pixel_per_metric = 0
lime_sizes = []
found_list = []
skip = 0
while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_rgb = frame
    frame_resized = cv2.resize(frame_rgb, (width, height))
    #frame_resized = cv2.resize(frame_rgb, (480, 320))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    
    if skip == 0:
        skip = 3
    # Perform the actual detection by running the model with the image as input
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        elapsed_time.append(time.time() - start_time) 

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        elapsed_time.append(time.time() - start_time) 
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


                # counting objects and measure diameter of lime
                if xmin < LINE2 and xmax > LINE1 and not entry:
                    entry = True
                    
                if entry and xmax <= LINE1:
                    entry = False

                    if(int(classes[i])+1 == 1):
                        lime_found = time.time() - start_total_time
                        try:
                            lime_count += 1
                            lime_diameter = ((xmax - xmin) + (ymax - ymin))  / (2 * pixel_per_metric)
                            lime_sizes.append(lime_diameter)
                            found_list.append(lime_found)
                            print(f'lime {lime_count} is found at {lime_found}, Diameter(size): {lime_diameter * 1000:.3f} mm')
                        except: 
                            # marker must came first for calculating pixel/metric
                            lime_count -= 1
                            marker_count += 1
                            total_marker_width += ((xmax - xmin) + (ymax - ymin)) / 2 
                            pixel_per_metric = (total_marker_width / marker_count) / MARKER_DIAMETER   

                    elif(int(classes[i])+1 == 2):
                        marker_count += 1
                        total_marker_width += ((xmax - xmin) + (ymax - ymin)) / 2
                        
                        pixel_per_metric = (total_marker_width / marker_count) / MARKER_DIAMETER
    else:
        skip = skip - 1

    

    # insert Lime Count information text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
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
        frame,
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
    
    cv2.line(frame, pt1, pt2, (0,0,255), 2)
    pt1 = ( LINE2, 0 ) 
    pt2 = ( LINE2, int(sqsize) ) 
    cv2.line(frame, pt1, pt2, (0,0,255), 2) 


    # All the results have been drawn on the frame, so it's time to display it.
    frame = cv2.resize(frame, (480, 320))
    cv2.imshow('TFLITE Lime detector', frame)

    key = cv2.waitKey(1)
    # Press 'q' to quit
    if key == ord('q'):
        break
    if key == 32: # Press spacebar to capture
        fname = IMG_PATH + 'lime_' + f'{idx:03}' + '.jpg'
        print('Saving to ' + fname)
        cv2.imwrite(fname, frame)
        idx = idx + 1

# Clean up
video.release()
cv2.destroyAllWindows()
total_time = time.time() - start_total_time
print('Total_time: ', total_time)
print('Average_elapsed_time: ', sum(elapsed_time[1:])/(len(elapsed_time) - 1))
