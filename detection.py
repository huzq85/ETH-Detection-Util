# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:08:01 2018

@author: huzq85
"""

import os
import cv2
import numpy as np
import tensorflow as tf

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

'''
Usage:
    python ped_detection.py --logtostderr \
    --inference_graph="${INFERENCE_GRAPH}"
    --label_map="${LABEL_MAP}"
    --test_image_folder="${TEST_IMAGE_FOLDER}"
'''

flags = tf.app.flags
flags.DEFINE_string('inference_graph', '', 'Path to the inference file')
flags.DEFINE_string('label_map','','Path to the label map file')
flags.DEFINE_string('test_image_folder','','Folder which contains the testing images')

FLAGS = flags.FLAGS


def main(_):
    assert FLAGS.inference_graph, '"Inference file missing!"'
    assert FLAGS.label_map, '"Label map file missing!"'
    assert FLAGS.test_image_folder, '"Testing image folder missing!"'
    
    test_imgs = []
    PATH_TO_CKPT = FLAGS.inference_graph
    PATH_TO_LABELS = FLAGS.label_map
    PATH_TO_TEST_FOLDER = FLAGS.test_image_folder
    PATH_TO_TEST_SAVE_FOLDER = os.path.join(os.path.dirname(PATH_TO_TEST_FOLDER),
                                            'Test_Result')
    if not os.path.exists(PATH_TO_TEST_SAVE_FOLDER):
        os.mkdir(PATH_TO_TEST_SAVE_FOLDER)
    
    for root, parent, files in os.walk(PATH_TO_TEST_FOLDER):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                img_path = os.path.join(root, file)
#                print(img_path)
                test_imgs.append(img_path)
    # Number of classes the object detector can identify
    NUM_CLASSES = 1
    
    # Load the label map.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)
    
    # Define input and output tensors (i.e. data) for the object detection classifier
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # Open a file to store the prediction results
    predict_file = open(os.path.join(PATH_TO_TEST_SAVE_FOLDER, 'Predict_Result.idl'), 'w')
    for test_img in test_imgs:
        image = cv2.imread(test_img)
        height, width, channel = image.shape
        image_expanded = np.expand_dims(image, axis=0)
        
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.50,
            )
        
        # To save the prediction into a file (Both of the detection boxes and scores)
        boxes_count = len(boxes[0])
        # Coordinate order return from boxes: [ymin, xmin, ymax, xmax]
        for i in range(boxes_count):
            ymin = boxes[0][i][0] * height
            xmin = boxes[0][i][1] * width
            ymax = boxes[0][i][2] * height
            xmax = boxes[0][i][3] * width
            score = scores[0][i]
            if not (ymin==0 and xmin==0 and ymax==0 and xmin==0) and score != 0:
                predict_res = test_img.split('\\')[-1] + ':' + str((xmin, ymin, xmax, ymax)) + ' Score:' + str(score)
                predict_res += '\n'
                predict_file.write(predict_res)
#        To save the testing image
        img_name = test_img.split('\\')[-1]
        testing_img = os.path.join(PATH_TO_TEST_SAVE_FOLDER, img_name)
        print('Saving test image: ', testing_img)
        cv2.imwrite(testing_img, image)
    predict_file.close()
    print('Testing Finish!')

if __name__ == '__main__':
    tf.app.run()