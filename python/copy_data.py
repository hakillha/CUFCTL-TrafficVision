import argparse

from modules.utils.general_utils import create_data_dir, copy_images, convert_xml
# from modules.utils.general_utils import gen_tfrecords

def parse_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--in_path',
						dest='in_path', help='')
	parser.add_argument('--out_path',
						dest='out_path', help='')
	return parser.parse_args()

args = parse_args()

# create_data_dir(args.out_path)
# copy_images(args.in_path, args.out_path)
# labels, classnum = convert_xml(args.in_path, args.out_path)
