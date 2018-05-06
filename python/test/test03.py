import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

reconstructed_images = []

# tfrecords_filename = '../../data/tfrecords/uadetrac_train.record'
tfrecords_filename = '/home/yingges/Desktop/Research/2018Q1/CUFCTL-TrafficVision/data/tfrecords/uadetrac_train.record'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    # height = int(example.features.feature['height']
    #                              .int64_list
    #                              .value[0])
    
    # width = int(example.features.feature['width']
    #                             .int64_list
    #                             .value[0])
    
    # img_string = (example.features.feature['image_raw']
    #                               .bytes_list
    #                               .value[0])
    
    # annotation_string = (example.features.feature['mask_raw']
    #                             .bytes_list
    #                             .value[0])

    img_string = (example.features.feature['image/encoded']
                                .bytes_list
                                .value[0])
    
    img = np.fromstring(img_string, dtype=np.uint8)
    print(img.shape)
    # plt.imshow(img)
    # plt.show()

    # reconstructed_img = img_1d.reshape((height, width, -1))
    
    # annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    
    # # Annotations don't have depth (3rd dimension)
    # reconstructed_annotation = annotation_1d.reshape((height, width))
    
    # reconstructed_images.append((reconstructed_img, reconstructed_annotation))