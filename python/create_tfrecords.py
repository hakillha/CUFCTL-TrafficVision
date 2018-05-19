import argparse

from modules.utils.data_utils import create_data_dir, copy_images, convert_xml
from modules.utils.data_utils import create_labelmap, create_trainval_set, gen_tfrecords

def parse_args():
	parser = argparse.ArgumentParser(description='Convert the UA-DETRAC dataset into tfrecords.')
	parser.add_argument('--in_path',
						help='The folder that contains \'Insight-MVT_Annotation_Train\'.')
	parser.add_argument('--out_path',
						default='data/tfrecords',
						help='Defaults to \'data/tfrecords\'')
	parser.add_argument('--label_map_path',
						default='data/ua_detrac_labelmap.pbtxt',
						help='Specify input label map file path. '
						'Defaults to \'data/ua_detrac_labelmap.pbtxt\'')
	parser.add_argument('--output_postfix', 
						default='',
						help='Identifier that differentiates record files generated for different purposes.')
	parser.add_argument('--occ_ratio_threshold', 
						default=0.4,
						help='Discard the bounding boxes with occlusion exceeding this ratio. '
						'Defaults to .4.')
	parser.add_argument('--sqrt_area_threshold', 
						default=70,
						help='Discard the bounding boxes with sqrt area larger than this number. '
						'Defaults to 70.')
	parser.add_argument('--train', 
						default=0.8,
						help='Training set split (e.g., 0.8). Must add up to 1 with the val split. '
						'Defaults to .8.')
	parser.add_argument('--val', 
						default=0.2,
						help='Validation set split (e.g., 0.2). Must add up to 1 with the train split. '
						'Defaults to .2.')
	return parser.parse_args()

args = parse_args()

# create_data_dir(args.out_path)
# copy_images(args.in_path, args.out_path)
# labels, classnum = convert_xml(args.in_path, args.out_path)
# labels = ['car', 'bus', 'van', 'others']
# create_labelmap(labels)
# create_trainval_set(args.in_path, train=0.8, val=0.2)

gen_tfrecords(args.in_path, args.out_path, output_postfix=args.output_postfix,
			  train=args.train, val=args.val, label_map_path=args.label_map_path,
			  occ_ratio_threshold=args.occ_ratio_threshold,
			  bb_sqrt_area_threshold=args.sqrt_area_threshold)