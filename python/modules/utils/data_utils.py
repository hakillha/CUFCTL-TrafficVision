import os, shutil
import tensorflow as tf
import hashlib

from lxml import etree
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from os.path import join as pjoin

def better_makedirs(path):
	"""
	Allow the existence of the dir being created.
	"""
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != os.errno.EEXIST:
			raise

def copy_images(in_path, out_path):
	for video in os.listdir(in_path):
		for frame in os.listdir(pjoin(in_path, video)):
			if frame.split('.')[-1] == 'jpg':
				file = pjoin(in_path, video, frame)
				shutil.copy(file, pjoin(out_path, 'Train', 'JPEGImages', video + '_' + frame))

def create_data_dir(out_path):
	better_makedirs(out_path)
	better_makedirs(pjoin(out_path, 'Train','Annotations'))
	better_makedirs(pjoin(out_path, 'Train','JPEGImages'))

def convert_xml(in_path, out_path):
	labels = []
	classnum = 0
	annotation_path = pjoin(in_path, 'DETRAC-Train-Annotations-XML')
	for file in os.listdir(annotation_path):
		parse_xml(pjoin(annotation_path, file))
	return labels, classnum

def parse_xml(file_path):
	# xml_tree = etree.parse(file_path)
	# root = xml_tree.getroot()

	# for child in root:
	# 	if child.tag == 'frame':
	# 		for detect in child[0]:
	pass

def create_labelmap(labels):
	f = open(pjoin('data', 'ua_detrac_labelmap.pbtxt'), 'w')
	for idx, label in enumerate(labels):
		line = 'item {\n'; f.write(line)
		line = '\tid: ' + str(idx+1) + '\n'; f.write(line)
		line = '\tname: "' + label + '"\n}\n'; f.write(line)
	f.close()

def create_trainval_set(in_path, train, val):
	pass
# 	n = train / val + 1
# 	f_train = open(pjoin('data', 'train.txt'), 'w')
# 	f_val = open(pjoin('data', 'val.txt'), 'w')
# 	for video in os.listdir(in_path):
# 		for idx, frame in enumerate(sorted(os.listdir(pjoin(in_path, video)))):
# 			frame_name = frame.split('.')[0]
# 			if idx % n == 0:
# 				f_val.write(frame_name + '\n')
# 			else:
# 				f_train.write(frame_name + '\n')
# 	f_train.close()
# 	f_val.close()

def gen_tfrecords(in_path, out_path='data/tfrecords', label_map_path='data/ua_detrac_labelmap.pbtxt', train=0.8, val=0.2, occ_ratio_threshold=0.4):
	better_makedirs(out_path)
	train_writer = tf.python_io.TFRecordWriter(out_path + '/uadetrac_train.record')
	val_writer = tf.python_io.TFRecordWriter(out_path + '/uadetrac_val.record')
	img_height = 540
	img_width = 960
	label_map_dict = label_map_util.get_label_map_dict(label_map_path)
	n = train / val + 1

	img_path1 = 'Insight-MVT_Annotation_Train'
	img_path2 = pjoin(in_path, img_path1)
	for video in os.listdir(img_path2):
		annotation_path = pjoin(in_path, 'DETRAC-Train-Annotations-XML') + '/' + video + '.xml'
		xml_tree = etree.parse(annotation_path)
		root = xml_tree.getroot()
		
		img_idx = 0
		img_path3 = pjoin(img_path2, video)
		image_list = sorted(os.listdir(img_path3))
		for child in root:
			if child.tag == 'frame':
				# check if the frame has no annotation?
				filename = video + '_' + image_list[img_idx]
				with tf.gfile.GFile(pjoin(img_path3, image_list[img_idx]), 'rb') as fid:
					encoded_jpg = fid.read()
				key = hashlib.sha256(encoded_jpg).hexdigest()
				
				xmin = []
				ymin = []
				xmax = []
				ymax = []
				classes_text = []
				classes = []
				truncated = []
				poses = []
				difficult_obj = []
				for detect in child[0]:
					bbox = detect[0]
					bb_width = float(bbox.attrib['width'])
					bb_height = float(bbox.attrib['height'])
					xmin.append(float(bbox.attrib['left']) / img_width)
					ymin.append(float(bbox.attrib['top']) / img_height)
					xmax.append((float(bbox.attrib['left']) + bb_width) / img_width)
					ymax.append((float(bbox.attrib['top']) + bb_height) / img_height)

					if len(detect) == 3:
						occ = detect[2][0]
						occ_ratio = float(occ.attrib['width']) * float(occ.attrib['height']) / (bb_width * bb_height )
						if occ_ratio > occ_ratio_threshold:
							continue

					att = detect[1]
					classes_text.append(att.attrib['vehicle_type'].encode('utf8'))
					classes.append(label_map_dict[att.attrib['vehicle_type']])
					difficult_obj.append(0)
					truncated.append(0)
					poses.append('Unspecified'.encode('utf8'))

				# type checking if error
				example = tf.train.Example(features=tf.train.Features(feature={
										'image/height': dataset_util.int64_feature(img_height),
										'image/width': dataset_util.int64_feature(img_width),
										'image/filename': dataset_util.bytes_feature(
										  filename.encode('utf8')),
										'image/source_id': dataset_util.bytes_feature(
										  filename.encode('utf8')),
										'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
										'image/encoded': dataset_util.bytes_feature(encoded_jpg),
										'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
										'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
										'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
										'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
										'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
										'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
										'image/object/class/label': dataset_util.int64_list_feature(classes),
										'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
										'image/object/truncated': dataset_util.int64_list_feature(truncated),
										'image/object/view': dataset_util.bytes_list_feature(poses),
										}))

				img_idx += 1
				
				if img_idx % n == 0:
					val_writer.write(example.SerializeToString())
				else:
					train_writer.write(example.SerializeToString())

	val_writer.close()
	train_writer.close()

# put n test for trainval division b4 writing
# check the data type b4 constructing tf.examples
# for frame in os.listdir(pjoin(in_path, video)):
			
