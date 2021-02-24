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

category_index = {1: {'id': 1, 'name': 'lime'}, 2: {'id': 2, 'name': 'marker'},}

@tf.function
def detect(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def detect_objects(img):
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    results = detect(input_tensor)

    bboxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = results['detection_scores'][0].numpy()

    return (bboxes, classes, scores)


def overlay_objects(img, bboxes, classes, scores):
    image = img.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image,
            bboxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.8,
            agnostic_mode=False)
    return image

def print_result(lime_count, marker_count, lime_list, total_time):
    print(f'lime count: {lime_count}')
    print(f'marker count: {marker_count}')
    print(f'lime size list: {lime_list}')
    print(f'Total time: {total_time:.2f} s')
    print('Finished')