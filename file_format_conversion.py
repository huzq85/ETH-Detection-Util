# -*- coding: utf-8 -*-
"""
Created on Mon May  7 06:35:03 2018

@author: huzq85
"""
import re
import json
import tensorflow as tf

'''
Usage:
    python file_format_conversion.py \
    --annotation_file="{$ANNOTATION_FILE}"
    --prediction_file="${PREDICT_FILE}"
'''

flags = tf.app.flags
flags.DEFINE_string('annotation_file','','To specify the original annotation file(*.idl)')
flags.DEFINE_string('prediction_file','','To specify a predict file(*.idl)')

FLAGS = flags.FLAGS

def convert_anno_file(origi_file, json_file):
    # Original idl format: "left/image_00000026.png": (186, 149, 251, 346):1, (464, 215, 488, 283):1
    # Return json format: {"img_00285.png": [[480, 457, 515, 529], [637, 435, 676, 536]]}
    
    anno_map = {}
    f_read = open(origi_file)
    lines = f_read.readlines()
    for line in lines:
        file_name_from_anno = line.split(':')[0] # '"left/image_00000986.png"'
        if file_name_from_anno.startswith('"') and file_name_from_anno.endswith('"'):
            file_name_from_anno = file_name_from_anno[1:-1] # To remove the '"' at the beginning and the end of the string
        if '/' in file_name_from_anno:
            file_name_from_anno = file_name_from_anno.split('/')[-1]
        annos_info = line[line.index(':')+1:-2] # annotation info for every image, 
        # drop the last character ';' or '.' in every line
        # [' (212, 209, 238, 270)', '-1, (233, 201, 260, 284)', '-1, (287, 215, 305, 260)', '-1']
        annos_info = annos_info.lstrip()
        #TODO: To investigate a potential bug following: if the annotation file
        # does not include an empty line at the bottom, the split of the annotation
        # may cause an exception.
        annos_info_list = re.compile("(?<!^)\s+(?=[,(])(?!.\s)").split(annos_info)
#        item_list = [list(item,) for item in annos_info_list]
#        print(item_list)
        single_anno = []
        for item in annos_info_list:
            tup_str = item.split(':')[0]
            if tup_str.startswith('('):
                tup_str = tup_str.lstrip('(')
            if tup_str.endswith(')'):
                tup_str = tup_str.strip(')')
            lis = tup_str.split(',')
            int_lis = [int(item) for item in lis]
            single_anno.append(int_lis)
        anno_map[file_name_from_anno] = single_anno
        # print(res for res in anno_map) # This line used for debug
    with open(json_file, 'w') as js:
        json.dump(anno_map, js)
#    return anno_map

def convert_predict_file(origi_file, json_file):
    # Original idl format: "left/image_00000026.png": (186, 149, 251, 346):1, (464, 215, 488, 283):1
    # Return format: {"img_00329.png": 
    #                   {"boxes": 
    #                       [
    #                            [429, 434, 534, 506], 
    #                            [342, 457, 413, 547], 
    #                            [422, 430, 443, 450]
    #                       ], 
    #                       "scores": 
    #                       [0.0505, 0.0634, 0.0636]
    #                   }
    #                }
    f_file = open(origi_file, 'r')
    lines = f_file.readlines()
    res_map = {}
    key = ''
    box_map = None
    score_map = None
    for line in lines:
        file_name = line.split(':')[0]
#        key = file_name
        score = float(line.split(':')[-1])
        line = line.split(':')[1] # Predict box
        boxes_str = line.rsplit(' ',1)[0]
        if boxes_str.startswith('('):
            boxes_str = boxes_str.lstrip('(')
        if boxes_str.endswith(')'):
            boxes_str = boxes_str.strip(')')
        boxes_str_lis = boxes_str.split(',')
        boxes_lis = [float(item) for item in boxes_str_lis]
        
        if key == file_name:
            box_map['boxes'].append(boxes_lis)
            score_map['scores'].append(score)
#            print(key, box_map, score_map)
        else:
            box_map = {}
            box_map['boxes'] = []
            score_map = {}
            score_map['scores'] = []
            key = file_name
            if key not in res_map.keys():
                box_map['boxes'].append(boxes_lis)
                score_map['scores'].append(score)
            res_map[key] = dict(list(box_map.items()) + list(score_map.items()))
 
#    print(res_map)
    with open(json_file, 'w') as js:
        json.dump(res_map, js)

# To make sure the ground truth file contains the same item with the prediction file
def process_anno_file(anno_file, predict_file, res_file):
    anno_read = open(anno_file, 'r')
    predict_read = open(predict_file, 'r')
    lines_anno = anno_read.readlines()
    lines_pre = predict_read.readlines()
    res_lines = []
    for line_pre in lines_pre:
        file_name_pre = line_pre.split(':')[0]
        for line_anno in lines_anno:
            file_name_anno = line_anno.split(':')[0]

            if file_name_pre in file_name_anno:
                res_lines.append(line_anno)
    print(res_lines)
    with open(res_file, 'w') as f_writer:
        f_writer.writelines(res_lines)
        
        

if __name__ == '__main__':
    assert FLAGS.annotation_file, 'Annotation(*.idl) file missing'
    assert FLAGS.prediction_file, 'Prediction file(*.idl) missing'
    
    anno_idl = FLAGS.annotation_file
    predict_idl = FLAGS.prediction_file
    anno_name = anno_idl.split('\\')[-1].split('.')[0]
    ground_truth_idl = anno_idl.replace(anno_name, 'gt_file')
    
    anno_json = anno_idl.replace('idl','json')
    pred_json = predict_idl.replace('idl','json')
    process_anno_file(anno_idl, predict_idl, ground_truth_idl)
    convert_anno_file(ground_truth_idl, anno_json)
    convert_predict_file(predict_idl, pred_json)
    print('Convert Done!')
    