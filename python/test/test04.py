import argparse
# import re
# # import tensorflow as tf

# # dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10, 8, 2]))
# # print(dataset1.output_types)  # ==> "tf.float32"
# # print(dataset1.output_shapes)  # ==> "(10,)"

# # dataset1 = tf.data.Dataset.from_tensors(tf.random_uniform([4, 10, 8, 2]))
# # print(dataset1.output_types)  # ==> "tf.float32"
# # print(dataset1.output_shapes)  # ==> "(10,)"

# def natural_sort(l):
# 	# match integers and convert to actual intergers
# 	int_value = lambda key: int(re.search('\d+', key).group())
# 	return sorted(l, key=int_value)

# x = ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']

# print(natural_sort(x))
# print(sorted(x))

def parse_args():
	parser = argparse.ArgumentParser(description='Convert the UA-DETRAC dataset into tfrecords.')
	parser.add_argument('--sqrt_area_threshold', 
						default=70,
						help='Discard the bounding boxes with sqrt area larger than this number. '
						'Defaults to 70.')
	return parser.parse_args()

args = parse_args()

print(80 < args.sqrt_area_threshold)