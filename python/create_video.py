import argparse
import cv2
import os

from modules.utils.misc_utils import natural_sort

def parse_args():
	parser = argparse.ArgumentParser(
		description='Concatenate the output detection images and create a video.')
	parser.add_argument('--image_dir',
						dest='image_dir', 
						help='The folder that contains output images for a video sequence.')
	parser.add_argument('--video_name',
						dest='video_name', 
						help='Output video name.')
	parser.add_argument('--frame_rate',
						type=int,
						default=20,
						dest='frame_rate', 
						help='Output video frame rate, should be identical with that of the original video.')
	parser.add_argument('--channel_transpose',
						# default='[2, 1, 0]', #
						dest='channel_transpose', 
						help='Swap the channels when input data don\'t come in as RGB order. '
						'Mostly useful when the inference is run on videos.')
	return parser.parse_args()



args = parse_args()
image_folder = args.image_dir
images = natural_sort([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
# the framerate depends on that of the original video
video = cv2.VideoWriter(args.video_name, cv2.VideoWriter_fourcc(*"MJPG"), args.frame_rate, (width, height))

# count = 0
for image in images:
	# print(image)
	input_im_path = os.path.join(image_folder, image)
	# print(input_im_path)
	if args.channel_transpose:
		video.write(cv2.imread(input_im_path)[:,:,eval(args.channel_transpose)])
	else:
		video.write(cv2.imread(input_im_path))
	# count += 1
# print(count)
cv2.destroyAllWindows()
video.release()