#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

class THEME:
 FLAGS = None
 def __init__(self):
  # Data Parameters
  tf.flags.DEFINE_string("data_file", "./data/20_newsgroups", "Data source for the positive data.")

  # Eval Parameters
  tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
  tf.flags.DEFINE_string("checkpoint_dir", "./snapshot/checkpoints/", "Checkpoint directory from training run")
  tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

  # Misc Parameters
  tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
  tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

  self.FLAGS = tf.flags.FLAGS
  self.FLAGS._parse_flags()

 def inference(self, note):
  # CHANGE THIS: Load data. Load your own data here
  if self.FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(self.FLAGS.data_file)
    y_test = np.argmax(y_test, axis=1)
  else:
    x_raw = []
    x_raw.extend(note)

  # Map data into vocabulary
  vocab_path = os.path.join('./snapshot/checkpoints', "..", "vocab")
  print(vocab_path)
  vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
  x_test = np.array(list(vocab_processor.transform(x_raw)))

  checkpoint_file = './snapshot/checkpoints/model-5900'
  graph = tf.Graph()
  with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), 64, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
  return all_predictions
if __name__=="__main__":
  THEME = THEME()
  cls_ID = THEME.inference("THIS IS MY NOTE, MY NAME IS SLEEP EARLY AND KEEP CODING.")[:1]
  print(cls_ID)
