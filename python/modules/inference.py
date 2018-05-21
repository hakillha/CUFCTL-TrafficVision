import cv2
import numpy as np
import os, re
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

def save_as_text_file(out, height, width, output_dict, imgname=None, count=None):
  if imgname:
    frame_count = int(re.search('\d+', imgname).group())
  elif count:
    frame_count = count
  for idx, bb in enumerate(output_dict['detection_boxes']):
    bbox = [i * j for i, j in zip(bb, [width, height, width, height])]
    detect_line = [frame_count, -1, bbox[0], bbox[1], bbox[2], bbox[3], 
                   output_dict['detection_scores'][idx], -1, -1]
    out.write(str(detect_line)[1:-1] + '\n')

def inference_on_frames(FLAGS, detection_graph, category_index):
  video_name = FLAGS.video_name
  video_dir = os.path.join(FLAGS.test_data_dir, video_name)
  test_img_paths = [os.path.join(video_dir, im) 
  					         for im in os.listdir(video_dir)]
  output_dir = os.path.join(FLAGS.output_dir, video_name)
  better_makedirs(output_dir)

  for imfile in test_img_paths:
    image = Image.open(imfile)
    image = load_image_into_numpy_array(image)
    output_dict = run_inference_for_single_image(image, detection_graph)
    if FLAGS.output_format == 'image':
      vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
      cv2.imwrite(output_dir + '/' + imfile.split('/')[-1].split('.')[0] + '.png', image)
    elif FLAGS.output_format == 'text':
      if not os.path.isfile(FLAGS.text_file_name):
        out = open(os.path.join(FLAGS.output_dir, FLAGS.text_file_name), 'w')
      save_as_text_file(out, image.shape[0], image.shape[1], output_dict, imgname=imfile)
  if FLAGS.output_format == 'text':
    out.close()

def inference_on_video(FLAGS, detection_graph, category_index):
  output_dir = os.path.join(FLAGS.output_dir, FLAGS.whole_video_path.split('/')[-1])
  better_makedirs(output_dir)

  vidcap = cv2.VideoCapture(FLAGS.whole_video_path)
  success, image = vidcap.read()
  count = 1
  while success:
    image = image[:,:,eval(FLAGS.channel_transpose)]
    output_dict = run_inference_for_single_image(image, detection_graph)
    if FLAGS.output_format == 'image':
      vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
      cv2.imwrite(output_dir + '/frame%d.png' % count, image)
    elif FLAGS.output_format == 'text':
      if not os.path.isfile(FLAGS.text_file_name):
        out = open(os.path.join(FLAGS.output_dir, FLAGS.text_file_name), 'w')
      save_as_text_file(out, image.shape[0], image.shape[1], output_dict, count=count)
    success, image = vidcap.read()
    count += 1
  if FLAGS.output_format == 'text':
    out.close()

def inference_on_video_test(FLAGS, detection_graph, category_index):
  output_dir = os.path.join(FLAGS.output_dir, FLAGS.whole_video_path.split('/')[-1])
  better_makedirs(output_dir)

  video_dir = 'data/input/SR20_AT_MOG_PRESET_5.avi'
  test_img_paths = [os.path.join(video_dir, im) 
            for im in os.listdir(video_dir)]

  vidcap = cv2.VideoCapture(FLAGS.whole_video_path)
  success, image = vidcap.read()
  count = 0
  for imfile in sorted(test_img_paths):
    image = image[:,:,[2,1,0]]

    image01 = Image.open(imfile)
    image01 = load_image_into_numpy_array(image01)
    print('*****')
    print(image[100,100,:])
    print(image01[100,100,:])

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(image)
    fig.add_subplot(1,2,2)
    plt.imshow(image01)
    plt.show()
    plt.close()

    success, image = vidcap.read()
    count += 1