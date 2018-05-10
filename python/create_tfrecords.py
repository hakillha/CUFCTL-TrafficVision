import argparse

from modules.utils.data_utils import create_data_dir, copy_images, convert_xml
from modules.utils.data_utils import create_labelmap, create_trainval_set, gen_tfrecords

def parse_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--in_path',
						dest='in_path', 
						help='The folder that contains \'Insight-MVT_Annotation_Train\'.')
	parser.add_argument('--out_path',
						default='data/tfrecords',
						dest='out_path', help='')
	parser.add_argument('--label_map_path',
						default='data/ua_detrac_labelmap.pbtxt',
						dest='label_map_path',
						help='Specify input label map file path.')
	parser.add_argument('--occ_ratio_threshold', 
						default=0.4,
						dest='occ_ratio_threshold', 
						help='Discard the bounding boxes with occlusion exceeding this ratio.')
	parser.add_argument('--train', 
						default=0.8,
						dest='train', 
						help='Training set split.')
	parser.add_argument('--val', 
						default=0.2,
						dest='val', 
						help='Validation set split.')
	return parser.parse_args()

args = parse_args()

# create_data_dir(args.out_path)
# copy_images(args.in_path, args.out_path)
# labels, classnum = convert_xml(args.in_path, args.out_path)
labels = ['car', 'bus', 'van', 'others']
create_labelmap(labels)
# create_trainval_set(args.in_path, train=0.8, val=0.2)

gen_tfrecords(args.in_path, train=args.train, val=args.val,
			  out_path=args.out_path, label_map_path=args.label_map_path,
			  occ_ratio_threshold=args.occ_ratio_threshold)