import time
import os
import glob
import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

pipeline_path = "output/pipeline.config"
checkpoint_path = "output/checkpoints"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(checkpoint_path, 'ckpt-1')).expect_partial()

@tf.function
def detect(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

elapsed_time = []
images_np = []

for img_file in glob.glob('data/images/*.jpg'):
    img = cv2.imread(img_file)
    start_time = time.time()
    input_tensor = tf.convert_to_tensor([img], dtype=tf.float32)
    results = detect(input_tensor)
    end_time = time.time()
    elapsed_time.append( (end_time - start_time) )
    bboxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = results['detection_scores'][0].numpy()
    print(img_file, classes[scores > 0.8], bboxes[scores > 0.8])

print(elapsed_time)



# video object detection
cap = cv2.VideoCapture('data/clips/test_clip.h264')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
sqsize = max(width,height)

category_index = {1: {'id': 1, 'name': 'lime'}, 2: {'id': 2, 'name': 'marker'},}

elapsed_time = []

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
    
    start_time = time.time()
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    results = detect(input_tensor)
    end_time = time.time()
    elapsed_time.append( (end_time - start_time) )
    bboxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = results['detection_scores'][0].numpy()

    img = frame.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            img,
            bboxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.8,
            agnostic_mode=False)

    cv2.imshow('Preview', img)     
    key = cv2.waitKey(int(1000/fps))
    if key == ord('q'):
        break
print(elapsed_time)