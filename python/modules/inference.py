import numpy as np
import os
import tensorflow as tf

from python.modules.utils.data_utils import better_makedirs

from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def inference_on_frames(FLAGS, detection_graph, category_index):
	video_name = FLAGS.video_name
	video_dir = os.path.join(FLAGS.test_data_dir, video_name)
	test_img_paths = [os.path.join(video_dir, im) 
						for im in os.listdir(video_dir)]
	# test_img_paths = []
	output_dir = os.path.join(FLAGS.output_dir, video_name)
	better_makedirs(output_dir)

	for imfile in test_img_paths:
		image = Image.open(imfile)
		image = load_image_into_numpy_array(image)
		output_dict = run_inference_for_single_image(image, detection_graph)
		vis_util.visualize_boxes_and_labels_on_image_array(
			image,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			category_index,
			instance_masks=output_dict.get('detection_masks'),
			use_normalized_coordinates=True,
			line_thickness=8)
		fig = plt.figure(figsize=(12, 8))
		plt.imshow(image)
		# plt.show()
		plt.savefig(output_dir + '/' + imfile.split('/')[-1].split('.')[0] + '.png',
					bbox_inches='tight')
		plt.close()

def inference_on_video(FLAGS, detection_graph, category_index):
  pass