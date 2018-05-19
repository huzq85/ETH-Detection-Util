# -*- coding: utf-8 -*-
"""
Created on Sun May  6 23:51:18 2018

@author: huzq85
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
Usage:
    python mAP_calculator.py --logtostderr \
    --ground_truth_json="${GT_JSON_FILE}"
    --predict_json="${PREDICT_JSON}"
'''


flags = tf.app.flags
flags.DEFINE_string('ground_truth_json', '','Path to the ground truth json file')
flags.DEFINE_string('predict_json', '','Path to the predicted json file')
FLAGS = flags.FLAGS

sns.set_style('white')
sns.set_context('poster')

COLORS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6','C7','C8','C9','C0']

def calc_iou_individual(pred_box, ground_truth_box):
    x1_t, y1_t, x2_t, y2_t = ground_truth_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
#    if (x1_t > x2_t) or (y1_t > y2_t):
#        raise AssertionError(
#            "Ground Truth box is malformed? true box: {}".format(ground_truth_box))
    if (x1_t > x2_t):
        x1_t, x2_t = x2_t, x1_t
    if (y1_t > y2_t):
        y1_t, y2_t = y2_t, y1_t

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(ground_truth_boxes, prediction_boxes, iou_threshold):
    all_pred_indices = range(len(prediction_boxes))
    all_ground_truth_indices = range(len(ground_truth_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(ground_truth_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_ground_truth_indices) == 0:
        tp = 0
        fp = len(prediction_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    ground_truth_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(prediction_boxes):
        for igb, ground_truth_box in enumerate(ground_truth_boxes):
            iou = calc_iou_individual(pred_box, ground_truth_box)
            if iou > iou_threshold:
                ground_truth_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(prediction_boxes)
        fn = len(ground_truth_boxes)
    else:
        ground_truth_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            ground_truth_idx = ground_truth_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (ground_truth_idx not in ground_truth_match_idx) and (pr_idx not in pred_match_idx):
                ground_truth_match_idx.append(ground_truth_idx)
                pred_match_idx.append(pr_idx)
        tp = len(ground_truth_match_idx)
        fp = len(prediction_boxes) - len(pred_match_idx)
        fn = len(ground_truth_boxes) - len(ground_truth_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def get_model_scores_map(prediction_boxes):
    model_scores_map = {}
    for img_id, val in prediction_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

def get_avg_precision_at_iou(ground_truth_boxes, prediction_boxes, iou_threshold=0.5):
    model_scores_map = get_model_scores_map(prediction_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in prediction_boxes.keys():
        arg_sort = np.argsort(prediction_boxes[img_id]['scores'])
        prediction_boxes[img_id]['scores'] = np.array(prediction_boxes[img_id]['scores'])[arg_sort].tolist()
        prediction_boxes[img_id]['boxes'] = np.array(prediction_boxes[img_id]['boxes'])[arg_sort].tolist()

    prediction_boxes_pruned = deepcopy(prediction_boxes)

    Precisions = []
    Recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = ground_truth_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            ground_truth_boxes_img = ground_truth_boxes[img_id]
            box_scores = prediction_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    prediction_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            prediction_boxes_pruned[img_id]['scores'] = prediction_boxes_pruned[img_id]['scores'][start_idx:]
            prediction_boxes_pruned[img_id]['boxes'] = prediction_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                ground_truth_boxes_img, prediction_boxes_pruned[img_id]['boxes'], iou_threshold)

        prec, rec = calc_precision_recall(img_results)
        Precisions.append(prec)
        Recalls.append(rec)
        model_thrs.append(model_score_thr)

    Precisions = np.array(Precisions)
    Recalls = np.array(Recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(Recalls >= recall_level).flatten()
            prec = max(Precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'Precisions': Precisions,
        'Recalls': Recalls,
        'model_thrs': model_thrs}


def plot_pr_curve(
    Precisions, Recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(Recalls, Precisions, label=label, s=10, color=color)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax

if __name__ == '__main__':
    assert FLAGS.ground_truth_json, '"Ground truth file missing"'
    assert FLAGS.predict_json, '"Predict result file missing"'
    with open(FLAGS.ground_truth_json) as infile:
        ground_truth_boxes = json.load(infile)
    with open(FLAGS.predict_json) as infile:
        prediction_boxes = json.load(infile)

    # Test for a single IoU threshold
    iou_threshold = 0.5
    start_time = time.time()
    data = get_avg_precision_at_iou(ground_truth_boxes, prediction_boxes, iou_threshold=iou_threshold)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('Average Precisions: {:.4f}'.format(data['avg_prec']))

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thresholds = []
    for idx, iou_threshold in enumerate(np.linspace(0.5, 0.95, 10)):
        data = get_avg_precision_at_iou(ground_truth_boxes, prediction_boxes, iou_threshold=iou_threshold)
        avg_precs.append(data['avg_prec'])
        iou_thresholds.append(iou_threshold)

        Precisions = data['Precisions']
        Recalls = data['Recalls']
        ax = plot_pr_curve(
            Precisions, Recalls, label='{:.2f}'.format(iou_threshold), color=COLORS[idx], ax=ax)

    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thresholds = [float('{:.4f}'.format(thr)) for thr in iou_thresholds]
    print('mAP: {:.2f}'.format(100*np.mean(avg_precs)))
    print('Average Precisions: ', avg_precs)
    print('IoU Thresholds:  ', iou_thresholds)
    plt.legend(loc='upper right', title='IoU Threshold', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.show()