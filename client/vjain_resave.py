
import logging
import time
import grpc
from grpc import RpcError

import tensorflow as tf

init = tf.global_variables_initializer()

with tf.Session() as sess:
  #sess.run(init)
  #tf.reset_default_graph()  
  imported_meta = tf.train.import_meta_graph("/data/vernica_data/CNN_Model_v6/checkpoint-100.meta")
  imported_meta.restore(sess, tf.train.latest_checkpoint('/data/vernica_data/CNN_Model_v6/')) 

  from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
  print_tensors_in_checkpoint_file(file_name='/data/vernica_data/CNN_Model_v6/checkpoint-100', tensor_name='', all_tensors=False)

  print("Model restored.")


  #export_path = '/data/mmanning/vjain-model-data_v6/'
  #builder = tf.saved_model.builder.SavedModelBuilder(export_path)


