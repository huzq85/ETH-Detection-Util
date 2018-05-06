# tfrecords-convert
gen_tf_records.py uses for generating the TF records based on giving images and annotation file from ETH pedestrian dataset (https://data.vision.ee.ethz.ch/cvl/aess/dataset/).

Usage:
    python gen_tf_records.py --logtostderr \
    --annotation_file="${ANNOTATION_FILE}"
    --image_directory="${IMAGE_DIRECTORY}"
	[--training_ratio="${TRAINING_RATIO}"]
Note: The last parameter --training_ratio is optional. If it specified, then the originial dataset will be divided into two parts, one used for training and the other for testing.
If it is not specified, then the whole dataset will be used for training.
