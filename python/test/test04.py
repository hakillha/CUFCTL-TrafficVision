import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10, 8, 2]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset1 = tf.data.Dataset.from_tensors(tf.random_uniform([4, 10, 8, 2]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"