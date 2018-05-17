# extract gt from tfrecords (test01.py), visualize them with TFODAPI
import argparse
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def parse_args():
	parser = argparse.ArgumentParser(description='Ground truths visualization.')
	parser.add_argument('--input_path',
						default='/media/yingges/TOSHIBA EXT/models/201805/data/tfrecords/uadetrac_val.record',
						dest='input_path')
	parser.add_argument('--label_map_path',
						default='data/ua_detrac_labelmap.pbtxt',
						dest='label_map_path')
	parser.add_argument('--classnum', 
						default=4,
						dest='classnum',
						help='Number of classes.')
	return parser.parse_args()

def read(serialized_example):
	record = tf.parse_single_example(serialized_example,
									features={
									'image/height': tf.FixedLenFeature([], dtype=tf.int64),
									'image/width': tf.FixedLenFeature([], dtype=tf.int64),
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
	# record['image'] = tf.decode_raw(record['image/encoded'], tf.uint8)
	record['image'] = tf.image.decode_jpeg(record['image/encoded'])
	record['image/object/bbox/xmin'] = tf.sparse_tensor_to_dense(record['image/object/bbox/xmin'])
	record['image/object/bbox/xmax'] = tf.sparse_tensor_to_dense(record['image/object/bbox/xmax'])
	record['image/object/bbox/ymin'] = tf.sparse_tensor_to_dense(record['image/object/bbox/ymin'])
	record['image/object/bbox/ymax'] = tf.sparse_tensor_to_dense(record['image/object/bbox/ymax'])
	record['image/object/class/text'] = tf.sparse_tensor_to_dense(record['image/object/class/text'], default_value='')
	record['image/object/class/label'] = tf.sparse_tensor_to_dense(record['image/object/class/label'])
	return record

args = parse_args()
label_map = label_map_util.load_labelmap(args.label_map_path)
categories = label_map_util.convert_label_map_to_categories(
				label_map, max_num_classes=args.classnum, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
dataset = tf.data.TFRecordDataset(args.input_path)
dataset = dataset.map(read)
iterator = dataset.make_one_shot_iterator()
next_ele = iterator.get_next()

with tf.Session() as sess:
	try:
		for _ in range(10):
			record_out = sess.run(next_ele)
			image = record_out['image'].reshape([record_out['image/height'], record_out['image/width'], 3])
			bb = np.stack((record_out['image/object/bbox/ymin'],
						   record_out['image/object/bbox/xmin'],
						   record_out['image/object/bbox/ymax'],
						   record_out['image/object/bbox/xmax']))
			print(bb.shape)
			score_dummy = np.ones(record_out['image/object/class/label'].shape)
			vis_util.visualize_boxes_and_labels_on_image_array(
				image,
				np.transpose(bb),
				record_out['image/object/class/label'],
				score_dummy,
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8)
			plt.figure(figsize=(12, 8))
			plt.imshow(record_out['image'])
			# plt.show()
			plt.close()
	except tf.errors.OutOfRangeError:
		pass
