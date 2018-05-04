import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_path = '/media/yingges/TOSHIBA EXT/models/201805/data/tfrecords/uadetrac_val.record'  # address to save the hdf5 file

with tf.Session() as sess:
    feature={
            'image/filename': tf.FixedLenFeature([], dtype=tf.string),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/truncated': tf.VarLenFeature(dtype=tf.int64),
            'image/object/view': tf.VarLenFeature(dtype=tf.int64)
            }
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    # automatically goes to next file when reaching the end of a file (queue of that file empty?)?
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    
    # Cast label data into int32
    tmp1 = features['image/object/class/label']
    print(tmp1)
    label = tf.cast(features['image/object/class/label'], tf.int32)
    print(label)
    # Reshape image data into the original shape
    # image = tf.reshape(image, [224, 224, 3])
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

        # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j+1)
            plt.imshow(img[j, ...])
            plt.title('cat' if lbl[j]==0 else 'dog')
        plt.show()
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()