import os
import cv2
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ['Person'] # List of string class name of bounding box (1 per box)
  classes = [-1] # List of integer class id of bounding box (1 per box)

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

# Reading properties for every image file accompany with the annotation information
def get_files(f_path):
    # Get all of the images
  for root, parent, files in os.walk(r'E:\5-DataSets\ETH\Setup1\BAHNHOF'):
      for file in files:
          if (file.endswith('png')):
              img_file = os.path.join(root, file)
              print('Reading: ' + img_file) # Get the image files
              img = cv2.imread(img_file)
              height = img.shape[0]
              width = img.shape[1]
          if (file.endswith('idl')):
              annotation_file = os.path.join(root, file)
              print(annotation_file) # Get the annotation file
              f_read = open(annotation_file)
              lines = f_read.readlines()
              for line in lines:
                  file_name_from_anno = line.split(':')[0] # "left/image_00000986.png"
                  anno_info = line.split(':')[1:] # annotation info for every image
                  
                  
                  
          
  # Process the annotation file
def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  
  examples = []
  
  # TODO(user): Write code to read in your dataset to examples variable

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()