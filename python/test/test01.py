import tensorflow as tf

def read(serialized_example):
	features = tf.parse_single_example(
										serialized_example,
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

	filename = features['image/filename']
	image = tf.decode_raw(features['image/encoded'], tf.uint8)
	xmin = features['image/object/bbox/xmin']
	xmax = features['image/object/bbox/xmax']
	ymin = features['image/object/bbox/ymin']
	ymax = features['image/object/bbox/ymax']

	return filename, image
	# return filename

filename = 'data/tfrecords/uadetrac_val.record'
# filename = '/media/yingges/TOSHIBA EXT/models/201805/data/tfrecords/uadetrac_val.record'
dataset = tf.data.TFRecordDataset(filename)
# (filename, image) = dataset.map(read)
dataset = dataset.map(read)
iterator = dataset.make_one_shot_iterator()
next_ele = iterator.get_next()

with tf.Session() as sess:
	for _ in range(10):
		filename_out, image_out = sess.run(next_ele)
		print(filename_out)

# print(filename)
# print(image)
# print(outdict[0])
# print(outdict[1])
# print(outdict)