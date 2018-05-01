import os
import re
import cv2
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = example[0] # Image height
  width = example[1] # Image width
  filename = example[2] # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = b'png' # b'jpeg' or b'png'

  xmins = example[3] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example[4] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = example[5] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example[6] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = example[7] # List of string class name of bounding box (1 per box)
  classes = example[8] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

# Reading and processing annotations
def read_annotations(f_path):
    anno_map = {}
    for root, parent, files in os.walk(f_path):
        for file in files:
            if (file.endswith('idl')):
                  annotation_file = os.path.join(root, file)
                  print(annotation_file) # Get the annotation file
                  f_read = open(annotation_file)
                  lines = f_read.readlines()
                  for line in lines:
                      file_name_from_anno = line.split(':')[0] # "left/image_00000986.png"
                      annos_info = line[line.index(':')+1:-2] # annotation info for every image, 
                      # drop the last character ';' or '.' in every line
                      # [' (212, 209, 238, 270)', '-1, (233, 201, 260, 284)', '-1, (287, 215, 305, 260)', '-1']
                      annos_info = annos_info.lstrip()
                      annos_info_list = re.compile("(?<!^)\s+(?=[,(])(?!.\s)").split(annos_info)
                      anno_map[file_name_from_anno] = annos_info_list
    print(res for res in anno_map) # This line used for debug
    return anno_map
            
# Reading properties for every image
def read_images(f_path):
    imgs_map = {}
    for root, parent, files in os.walk(f_path):
        for file in files:
            if (file.endswith('png')):
                img_file = os.path.join(root, file)
                print('Reading: ' + img_file)
                img = cv2.imread(img_file)
                 # height <-- img.shape[0]
                 # width <-- img.shape[1]
                imgs_map[file] = img.shape
    print(res for res in imgs_map)
    return imgs_map

#  Making examples
# [480, 640, '"left/image_00000003.png"', 450, 477, 212, 278, 'Person', '-1']
def make_examples(f_path):
    examples = []
    imgs = read_images(f_path)
    annos = read_annotations(f_path)
    for anno_key in annos.keys():
        # print(anno_key)
        anno_key_sp = anno_key.split('/')[1]
        anno_key_sp = anno_key_sp.split('.')[0]
        # print(anno_key_sp)
        for img_key in imgs.keys():
            if anno_key_sp in img_key:
                #print(annos[anno_key])
                #print(imgs[img_key])
                annos_list = annos[anno_key]
                imgs_tuple = imgs[img_key]
                print(annos_list)
                print(imgs_tuple)
                for single_anno in annos_list:
                    single_instance = []
                    # Process coordinates
                    coordinate_str = single_anno.split(':')[0]
                    coordinate_tuple = eval(coordinate_str)
                    xmin = coordinate_tuple[0]
                    ymin = coordinate_tuple[1]
                    xmax = coordinate_tuple[2]
                    ymax = coordinate_tuple[3]
                    class_label = single_anno.split(':')[1].strip(',')
                    single_instance.append(imgs_tuple[0])
                    single_instance.append(imgs_tuple[1])
                    single_instance.append(anno_key)
                    # May append endcoded_image_data
                    single_instance.append(xmin)
                    single_instance.append(xmax)
                    single_instance.append(ymin)
                    single_instance.append(ymax)
                    single_instance.append('Person')
                    single_instance.append(class_label)
                    examples.append(single_instance)
    return examples                

          
  # Process the annotation file
def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  
  # TODO(user): Write code to read in your dataset to examples variable
  examples = make_examples(r'F:\81-DataSets\BAHNHOF')

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()