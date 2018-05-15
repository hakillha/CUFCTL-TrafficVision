import os
import tensorflow as tf

from python.modules.inference import inference_on_frames, inference_on_video

from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('model_dir',
					'data/checkpoints/ssd_mobilenet_v2_inference_06',
					'Frozen graph directory.')
flags.DEFINE_string('labelmap_path',
					'data/ua_detrac_labelmap.pbtxt',
					'Labelmap path. Defaults to \'data/ua_detrac_labelmap.pbtxt\'')
flags.DEFINE_string('test_data_dir',
					'/media/yingges/TOSHIBA EXT/datasets/trafficvision/UADETRAC/Insight-MVT_Annotation_Test',
					'Test images dir.')
flags.DEFINE_string('video_name',
					'MVI_40714',
					'Which video sequence to run inference on.')
flags.DEFINE_string('whole_video_path',
					None,
					# '/media/yingges/TOSHIBA EXT/datasets/DOT/traffic_video_samples/SR20_AT_MOG_PRESET_5.avi',
					'The path of the undivided video to run inference on.')
flags.DEFINE_string('output_dir',
					'data/inference_output',
					'Defaults to \'data/inference_output\'')
flags.DEFINE_integer('classnum', 
					 4,
					 'Number of classes.')
FLAGS = flags.FLAGS

def main(_):
	fg_path = os.path.join(FLAGS.model_dir, 'frozen_inference_graph.pb')
	print(fg_path)
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(fg_path, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	label_map = label_map_util.load_labelmap(FLAGS.labelmap_path)
	categories = label_map_util.convert_label_map_to_categories(
					label_map, max_num_classes=FLAGS.classnum, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	if FLAGS.whole_video_path:
		inference_on_video(FLAGS, detection_graph, category_index)
	else:
		inference_on_frames(FLAGS, detection_graph, category_index)

if __name__ == '__main__':
	tf.app.run()