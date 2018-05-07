# -*- coding: utf-8 -*-
"""
Created on Mon May  7 06:35:03 2018

@author: huzq85
"""
import re
import json

def convert_anno_file(origi_file, json_file):
    # Return format: {"img_00285.png": [[480, 457, 515, 529], [637, 435, 676, 536]]}
    
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
    # Return format: {"img_00329.png": {"boxes": [[429, 434, 534, 506], [342, 457, 413, 547], [422, 430, 443, 450], [357, 457, 384, 484], [654, 453, 687, 528], [430, 332, 484, 451], [523, 448, 541, 463], [525, 441, 571, 495], [406, 428, 478, 505], [420, 432, 459, 500], [420, 432, 449, 461], [670, 490, 693, 533], [670, 455, 689, 532], [546, 434, 572, 501]], "scores": [0.0505, 0.0634, 0.0636, 0.0661, 0.0716, 0.1086, 0.1169, 0.1316, 0.1328, 0.2834, 0.2942, 0.3387, 0.965, 0.9891]}}
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
    anno_file = r'F:\Test-Train\refined.idl'
    predict_file = r'F:\Test-Train\Predict_Result.idl'
    anno_json = r'F:\Test-Train\anno_json.json'
    pred_json = r'F:\Test-Train\pred_json.json'
    ground_truth_file = r'F:\Test-Train\gt_anno.idl'
    process_anno_file(anno_file, predict_file, ground_truth_file)
    convert_anno_file(ground_truth_file, anno_json)
    convert_predict_file(predict_file, pred_json)
    print('Convert Done!')
    