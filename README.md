# tfrecords-convert
gen_tf_records.py uses for generating the TF records based on giving images and annotation file from ETH pedestrian dataset (https://data.vision.ee.ethz.ch/cvl/aess/dataset/).

Usage:
    python gen_tf_records.py --logtostderr \
    --annotation_file="${ANNOTATION_FILE}"
    --image_directory="${IMAGE_DIRECTORY}"
