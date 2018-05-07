# ETH-Detection-Util
This utility contains pieces of code to do following things:

gen_tf_records.py uses for generating the TF records based on giving images and annotation file from ETH pedestrian dataset (https://data.vision.ee.ethz.ch/cvl/aess/dataset/).

ped_detection.py for predicting in the testing images based on the training model

file_format_convertion.py for converting idl files to json format

mAP_calculator.py for calculating the mAP scores based on the prediction results and the ground truth
