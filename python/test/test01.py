filename = 'data/tfrecords/uadetrac_val.record'
dataset = tf.data.TFRecordDataset(filename)

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
										'image/object/view': tf.VarLenFeature(dtype=tf.int64)
										})
	
	image = tf.decode_raw(features['image/filename'], tf.string)
	image = tf.decode_raw(features['image/encoded'], tf.uint8)
	image = tf.decode_raw(features['image/object/bbox/xmin'], tf.float32)
	image = tf.decode_raw(features['image/object/bbox/xmax'], tf.float32)
	image = tf.decode_raw(features['image/object/bbox/ymin'], tf.float32)
	image = tf.decode_raw(features['image/object/bbox/ymax'], tf.float32)