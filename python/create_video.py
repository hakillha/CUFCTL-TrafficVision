import cv2
import os

def parse_args():
	parser = argparse.ArgumentParser(
		description='Concatenate the output detection images and create a video.')
	parser.add_argument('--image_dir',
						dest='image_dir', 
						help='The folder that contains output images for a video sequence.')
	parser.add_argument('--video_name',
						dest='video_name', 
						help='Output video name.')
	return parser.parse_args()

args = parse_args()
image_folder = args.image_dir
video_name = args.video_name

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height))

for image in images:
	# print(image)
	video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()