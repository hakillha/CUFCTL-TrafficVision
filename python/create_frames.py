import argparse
import cv2

def parse_args():
	parser = argparse.ArgumentParser(description='Devide.')
	parser.add_argument('--in_path',
						dest='in_path', 
						help='The input video path.')
	parser.add_argument('--out_dir',
						default='data/input',
						dest='out_dir', 
						help='Defaults to \'data/input\'.')
	return parser.parse_args()

args = parse_args()
vidcap = cv2.VideoCapture(args.in_path)
success, image = vidcap.read()
count = 0
# success
# while success:
# 	cv.imwrite(args.out_dir + 'frame%d.png' % count, image)
# 	success, image = 