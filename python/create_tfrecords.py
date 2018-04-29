import argparse

from modules.utils.data_utils import create_data_dir, copy_images, convert_xml
from modules.utils.data_utils import create_labelmap, create_trainval_set, gen_tfrecords

def parse_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--in_path',
						dest='in_path', help='')
	parser.add_argument('--out_path',
						dest='out_path', help='')
	parser.add_argument('--label_map_path',
						dest='label_map_path', help='')
	parser.add_argument('--occ_ratio_threshold', type=float,
						dest='occ_ratio_threshold', help='')
	return parser.parse_args()

args = parse_args()

# create_data_dir(args.out_path)
# copy_images(args.in_path, args.out_path)
# labels, classnum = convert_xml(args.in_path, args.out_path)
labels = ['car', 'bus', 'van', 'others']
create_labelmap(labels)
# create_trainval_set(args.in_path, train=0.8, val=0.2)

#
gen_tfrecords(args.in_path, train=0.8, val=0.2, occ_ratio_threshold=args.occ_ratio_threshold)