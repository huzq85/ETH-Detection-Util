import os
import re
import io
import PIL.Image
import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# Read annotation file first, and then read the image according to the
# annotation file.
def create_tf_example(image_dir, file_name, anno_val):
    '''
    Args:
        anno_key: Single image file name
        anno_val: Annotation for the file
    '''
    # TODO(user): Populate the following variables from your example.
    # Need to read the images
    # print(file_name, anno_val)
    encoded_image_data = None
    image_path = os.path.join(image_dir, file_name)
    print(image_path)
    with tf.gfile.GFile(image_path, 'rb') as read_image:
        encoded_image_data = read_image.read()
    encode_img_io = io.BytesIO(encoded_image_data)
    img = PIL.Image.open(encode_img_io)
    img = np.asarray(img)
    
    height = int(img.shape[0]) # Image height
    width = int(img.shape[1]) # Image width
    filename = file_name # Filename of the image. Empty if image is not from file
    image_format = 'png'.encode('utf8') # b'jpeg' or b'png'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    classes_text = []
    
    for anno_item in anno_val:
        coordinate_str = anno_item.split(':')[0]
        coordinate_tuple = eval(coordinate_str)
        class_cate = int(anno_item.split(':')[1].strip(','))
    

        xmins.append(coordinate_tuple[0]/width) # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs.append(coordinate_tuple[2]/width) # List of normalized right x coordinates in bounding box
             # (1 per box)
        ymins.append(coordinate_tuple[1]/height) # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs.append(coordinate_tuple[3]/height) # List of normalized bottom y coordinates in bounding box
             # (1 per box)
        classes_text.append('Person'.encode('utf8')) # List of string class name of bounding box (1 per box)
        classes.append(class_cate) # List of integer class id of bounding box (1 per box)
    print(height, width, xmins, xmaxs, ymins, ymaxs, classes_text, classes)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
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
    print(f_path) # Get the annotation file
    f_read = open(f_path)
    lines = f_read.readlines()
    for line in lines:
        file_name_from_anno = line.split(':')[0] # '"left/image_00000986.png"'
        file_name_from_anno = file_name_from_anno[1:-1] # To remove the '"' at the beginning and the end of the string
        annos_info = line[line.index(':')+1:-2] # annotation info for every image, 
        # drop the last character ';' or '.' in every line
        # [' (212, 209, 238, 270)', '-1, (233, 201, 260, 284)', '-1, (287, 215, 305, 260)', '-1']
        annos_info = annos_info.lstrip()
        annos_info_list = re.compile("(?<!^)\s+(?=[,(])(?!.\s)").split(annos_info)
        anno_map[file_name_from_anno] = annos_info_list
        # print(res for res in anno_map) # This line used for debug
    return anno_map

        
  # Process the annotation file
def main(_):
    train_output_path = os.path.join(FLAGS.output_path, 'train.record')
    writer = tf.python_io.TFRecordWriter(train_output_path)
  
    # TODO(user): Write code to read in your dataset to examples variable
    anno_list = read_annotations(r'F:\81-DataSets\BAHNHOF\refined.idl')

    for key, val in anno_list.items():
        tf_example = create_tf_example(r'F:\81-DataSets\BAHNHOF\image', key, val)
        print(tf_example)
        writer.write(tf_example.SerializeToString())
    writer.close()
    
    tf.logging.info('Finished writing')


if __name__ == '__main__':
  tf.app.run()