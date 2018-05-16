# extract gt from tfrecords (test01.py), visualize them with TFODAPI
import argparse
import numpy as np
import tensorflow as tf

from object_detection.utils import visualization_utils as vis_util

def parse_args():
	parser = argparse.ArgumentParser(description='Ground truths visualization.')
	parser.add_argument('--input_path',
						default='/media/yingges/TOSHIBA EXT/models/201805/data/tfrecords/uadetrac_val.record',
						dest='input_path')
	parser.add_argument('--label_map_path',
						default='data/ua_detrac_labelmap.pbtxt',
						dest='label_map_path')
	return parser.parse_args()

def read(serialized_example):
	record = tf.parse_single_example(serialized_example,
									features={
									'image/filename': tf.FixedLenFeature([], dtype=tf.string),
									'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
									'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
									'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
									'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
									'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
									'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
									'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
									'image/object/difficult': tf.VarLenFeature(dtype=tf.int64),
									'image/object/truncated': tf.VarLenFeature(dtype=tf.int64),
									'image/object/view': tf.VarLenFeature(dtype=tf.string)
									})
	record['image'] = tf.decode_raw(record['image/encoded'], tf.uint8)
	return record

args = parse_args()
dataset = tf.data.TFRecordDataset(args.input_path)
dataset = dataset.map(read)
iterator = dataset.make_one_shot_iterator()
next_ele = iterator.get_next()

with tf.Session() as sess:
	try:
		for _ in range(10):
		# while True:
			record_out = sess.run(next_ele)
			print(record_out['image'].shape)
			# vis_util.visualize_boxes_and_labels_on_image_array(
			# 	image,
			# 	output_dict['detection_boxes'],
			# 	output_dict['detection_classes'],
			# 	output_dict['detection_scores'],
			# 	category_index,
			# 	instance_masks=output_dict.get('detection_masks'),
			# 	use_normalized_coordinates=True,
			# 	line_thickness=8)
	except tf.errors.OutOfRangeError:
		pass
