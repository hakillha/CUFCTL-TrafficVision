# extract gt from tfrecords (test01.py), visualize them with TFODAPI
import argparse, cv2, math, os
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from modules.utils.data_utils import better_makedirs

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
	parser.add_argument('--out_dir',
						default='data/input/stats/ua_detrac',
						dest='out_dir')
	parser.add_argument('--classnum', 
						type=int,
						default=4,
						dest='classnum',
						help='Number of classes.')
	parser.add_argument('--folder_size', 
						type=int,
						default=500,
						dest='folder_size',
						help='Keep size of subfolders in check for easy viewing.')
	parser.add_argument('--mode', 
						default='{"scale_hist": False, "gt_vis": True}',
						help='Specify the processing you would like to run.')
	parser.add_argument('--scale_data_file', 
						default='data/input/stats/ua_detrac/scale_data.npy',
						help='Data file for drawing scale histogram.')
	parser.add_argument('--vis_scale_threshold', 
						type=float,
						default=100000,
						help='Data file for drawing scale histogram.')
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
	record['image'] = tf.image.decode_jpeg(record['image/encoded'])
	record['image/object/bbox/xmin'] = tf.sparse_tensor_to_dense(record['image/object/bbox/xmin'])
	record['image/object/bbox/xmax'] = tf.sparse_tensor_to_dense(record['image/object/bbox/xmax'])
	record['image/object/bbox/ymin'] = tf.sparse_tensor_to_dense(record['image/object/bbox/ymin'])
	record['image/object/bbox/ymax'] = tf.sparse_tensor_to_dense(record['image/object/bbox/ymax'])
	record['image/object/class/text'] = tf.sparse_tensor_to_dense(record['image/object/class/text'], default_value='')
	record['image/object/class/label'] = tf.sparse_tensor_to_dense(record['image/object/class/label'])
	return record

def stats_parse(args, next_ele, category_index):
	mode = eval(args.mode)
	scale_data_file = args.scale_data_file
	count = 0
	bb_sqrt_area = []
	if mode['scale_hist'] and os.path.isfile(scale_data_file):
		scale_data = np.load(scale_data_file)
		plt.hist(scale_data, int(max(scale_data) / 20), ec='black')
		plt.show()

	with tf.Session() as sess:
		try:
			while True:
			# for _ in range(10):
				better_makedirs(args.out_dir)

				record = sess.run(next_ele)

				height = record['image/height']
				width = record['image/width']
				bb = np.transpose(np.stack((record['image/object/bbox/ymin'],
											record['image/object/bbox/xmin'],
											record['image/object/bbox/ymax'],
											record['image/object/bbox/xmax'])))

				keep_rows = []
				for idx, bbox in enumerate(bb):
					sqrt_area = math.sqrt((bbox[2] - bbox[0]) * height * (bbox[3] - bbox[1]) * width)
					if sqrt_area < args.vis_scale_threshold:
						keep_rows.append(idx)
					bb_sqrt_area.append(sqrt_area)
				bb = bb[keep_rows,:]

				# can skip images with no bb
				if mode['gt_vis']:
					if count % args.folder_size == 0:
						new_folder = os.path.join(args.out_dir, 'gt_vis', str(count))
						better_makedirs(new_folder)

					image = record['image'].reshape([height, width, 3])
					# print(bb.shape)
					score_dummy = np.ones(record['image/object/class/label'].shape)
					vis_util.visualize_boxes_and_labels_on_image_array(
						image,
						bb,
						record['image/object/class/label'],
						score_dummy,
						category_index,
						use_normalized_coordinates=True,
						line_thickness=8)
					# plt.figure(figsize=(12, 8))
					# plt.imshow(record['image'])
					# plt.show()
					# plt.close()
					cv2.imwrite(new_folder + '/' + record['image/filename'].split('.')[0] + '_image%d.png' % count, image[:,:,[2,1,0]])

				count += 1
		except tf.errors.OutOfRangeError:
			pass

	if mode['scale_hist']:
		np.save(scale_data_file, np.array(bb_sqrt_area))

args = parse_args()
label_map = label_map_util.load_labelmap(args.label_map_path)
categories = label_map_util.convert_label_map_to_categories(
				label_map, max_num_classes=args.classnum, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
dataset = tf.data.TFRecordDataset(args.input_path)
dataset = dataset.map(read)
iterator = dataset.make_one_shot_iterator()
next_ele = iterator.get_next()

stats_parse(args, next_ele, category_index)
