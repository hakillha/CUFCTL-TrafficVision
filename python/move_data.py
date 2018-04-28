import argparse

from modules.utils.general_utils import create_data_dir

def parse_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--in_path',
						dest='in_path', help='')
	parser.add_argument('--out_path',
						dest='out_path', help='')
	return parser.parse_args()

args = parse_args()