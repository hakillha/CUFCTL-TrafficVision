import argparse
import cv2
import os

from modules.utils.data_utils import better_makedirs

def parse_args():
	parser = argparse.ArgumentParser(description='Extract frames from a video.')
	parser.add_argument('--in_path',
						default='/media/yingges/TOSHIBA EXT/datasets/DOT/traffic_video_samples/SR20_AT_MOG_PRESET_5.avi',
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
outdir = os.path.join(args.out_dir, args.in_path.split('/')[-1])
better_makedirs(outdir)
while success:
	cv.imwrite(outdir + '/frame%d.png' % count, image)
	success, image = vidcap.read()
	count += 1