from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join


from random import shuffle
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.abspath("../vggish"))
import vggish_input
import vggish_postprocess
import vggish_params
import vggish_slim


flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'epochs', 1,
    'Number of times to loop through entire data set.')

flags.DEFINE_integer(
    'batch_size', 4,
    'Size of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')


flags.DEFINE_float('test_size', 0.2, 'Size of validation set as chunk of batch')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'pca_params', '../model/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'checkpoint', '../model/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_NUM_CLASSES = 2

def get_all_examples(path):
    input_examples =[]
    input_labels = []

    wavfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith("wav")]
    for wav in wavfiles:
        print("Processing", wav)
        try:
            signal_example = vggish_input.wavfile_to_examples(join(path, wav))
        except:
            print("Skipping {}, couldn't process".format(wav))
            continue
        print(type(signal_example))
        print(signal_example.shape)

        # Build own one-hot encoder 
        encoded = np.zeros((_NUM_CLASSES))
        if wav.startswith("cat"):
            encoded[0]=1
        else:
            encoded[1]=1
        encoded=encoded.tolist()

        #  # Encode each frame of the example, which results in the final label for this file
        signal_label =np.array([encoded]*signal_example.shape[0])    
        print("Printing signal label:")
        print(signal_label)

        # Check if a clean label can be extracted
        if signal_label != []:
            input_labels.append(signal_label)
            input_examples.append(signal_example)
        else:
          print("Skipping {}, unable extract clean signal label".format(wav))
          #  log.warn("Skipping {}, unable extract clean signal label".format(fname))
          continue
    return (input_examples, input_labels)

def get_random_epoch(input_examples, input_labels):
    """Shuffles up read-in examples and labels.
    The input audio files and the corresponding one-hot encoded labels of their audio frames are first 
    paired up, then shuffled and seperated again. Shuffling is done to prevent a common pattern due to
    reading in audio files in the same order each time and improve the model's ability to generalize.

    Args:
      input_examples (list): A list of 3-D np.arrays of shape [num_example, num_frames, num_bands]
      input_labels (list): A list 2-D np.arrays of shape [encoded_label, num_classes] where each
        example will consist n encoded labels, with n being the number of audio frames the example
        consists of.
      log: A Python logging object

    Returns:
      features (list): A shuffled list of input examples.
      labels (list): A shuffled list of input labels.
    """

    # Create a 3-D np.array of [sum(num_example), num_frames, num_bands]
    all_examples = np.concatenate([x for x in input_examples])
    
    # Create a 2-D np.array of [sum(encoded_labels), num_classes]
    all_labels = np.concatenate([x for x in input_labels])  
    
    # Pair up examples with corresponding labels in a list, shuffle it
    labeled_examples = list(zip(all_examples,all_labels))
    shuffle(labeled_examples)
    
    # Separate the shuffled list return the features and labels individually
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    
    return (features, labels)



def main(_):
  with tf.Graph().as_default(), tf.Session() as sess:
    # Define VGGish.
    embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):
      # Add a fully connected layer with 100 units.
      num_units = 100
      fc = slim.fully_connected(embeddings, num_units)

      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      logits = slim.fully_connected(
          fc, _NUM_CLASSES, activation_fn=None, scope='logits')
      tf.sigmoid(logits, name='prediction')

      # Add training ops.
      with tf.variable_scope('train'):
        global_step = tf.Variable(
            0, name='global_step', trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.GLOBAL_STEP])

        # Labels are assumed to be fed as a batch multi-hot vectors, with
        # a 1 in the position of each positive class label, and 0 elsewhere.
        labels = tf.placeholder(
            tf.float32, shape=(None, _NUM_CLASSES), name='labels')

        # Cross-entropy label loss.
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xent')
        loss = tf.reduce_mean(xent, name='loss_op')
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=vggish_params.LEARNING_RATE,
            epsilon=vggish_params.ADAM_EPSILON)
        optimizer.minimize(loss, global_step=global_step, name='train_op')

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

    # Locate all the tensors and ops we need for the training loop.
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
    global_step_tensor = sess.graph.get_tensor_by_name(
        'mymodel/train/global_step:0')
    loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
    train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

    # Load all Kaggle Cats and Dog wav files as frames with features.
    # 1 frame corresponds to 1 second of wav audio
    wav_path = "/Users/Colton.Saska@ibm.com/Documents/ScalableMachineLearning/scm_final_project/data/audio-cats-and-dogs/cats_dogs"
    all_examples, all_labels = get_all_examples(wav_path)

    # Create training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_validation, y_train, y_validation= train_test_split(all_examples, all_labels, test_size=FLAGS.test_size)

    # The training loop.
    for _ in range(FLAGS.epochs):
        # Shuffle X_train so that each epoch is randomized
        (features, labels) = get_random_epoch(X_train, y_train)

        # Break epoch into batches
        for i in range(0, len(features), FLAGS.batch_size):
            # TODO(csaska): break epoch into batches and train on batches
            pass

        [num_steps, loss, _] = sess.run(
            [global_step_tensor, loss_tensor, train_op],
            feed_dict={features_tensor: features, labels_tensor: labels})
        print('Step %d: loss %g' % (num_steps, loss))

if __name__ == '__main__':
  tf.app.run()