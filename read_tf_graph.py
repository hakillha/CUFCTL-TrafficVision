import tensorflow as tf
import sys

if len(sys.argv) != 2:
   print ("Usage ./read_tf_graph.py path_to_model")
   exit()

PATH_TO_MODEL = sys.argv[1]

detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)

for node in od_graph_def.node:
  print (node.name)

