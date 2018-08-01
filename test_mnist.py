import tensorflow as tf
import embedder
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
BATCH_SIZE = 64

#RM: Removed magic nums
LAYER1_FEATURES = 32
LAYER2_FEATURES = 64
CONV_WINDOW_DIM = 5     # 5 x 5 square used for the 2D convolution
DOWN_SAMPLING_FACTOR = 4
FULLY_CONNECTED_LAYER1_SIZE = 512
FULLY_CONNECTED_LAYER2_SIZE = NUM_LABELS

test_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(test_path, 'embedding')):
    os.makedirs(os.path.join(test_path, 'embedding'))


# 1. load model graph
def model():
    input_placeholder = tf.placeholder(tf.float32, \
                                       shape=(BATCH_SIZE, \
                                              IMAGE_SIZE, \
                                              IMAGE_SIZE, \
                                              NUM_CHANNELS))

    #########################################################################
    #INITIALIZE WEIGHTS & BIASES for the two convolution layers and the 
    #two fully connected layer
    conv1_weights = tf.Variable(tf.truncated_normal([CONV_WINDOW_DIM, \
                                                         CONV_WINDOW_DIM, \
                                                         NUM_CHANNELS, \
                                                         LAYER1_FEATURES], \
                                                    stddev=0.1, \
                                                    dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([LAYER1_FEATURES], \
                                dtype=tf.float32))
    
    conv2_weights = tf.Variable(tf.truncated_normal([CONV_WINDOW_DIM, \
                                                         CONV_WINDOW_DIM, \
                                                         LAYER1_FEATURES, \
                                                         LAYER2_FEATURES], \
                                                    stddev=0.1, \
                                                    dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[LAYER2_FEATURES], \
                                           dtype=tf.float32))
    
    fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // DOWN_SAMPLING_FACTOR * \
                                                   IMAGE_SIZE // DOWN_SAMPLING_FACTOR * \
                                                   LAYER2_FEATURES, \
                                                   FULLY_CONNECTED_LAYER1_SIZE], \
                                                stddev=0.1, \
                                                dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, \
                                         shape=[FULLY_CONNECTED_LAYER1_SIZE], \
                                         dtype=tf.float32))
    
    fc2_weights = tf.Variable(tf.truncated_normal([FULLY_CONNECTED_LAYER1_SIZE, \
                                                   FULLY_CONNECTED_LAYER2_SIZE], \
                                                  stddev=0.1, \
                                                  dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(0.1, \
                                         shape=[FULLY_CONNECTED_LAYER2_SIZE], \
                                         dtype=tf.float32))
    #########################################################################
    #CONVOLUTION + FULLY CONNECTED
    #
    #1st Convolution
    conv = tf.nn.conv2d(input_placeholder, conv1_weights, \
                        strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], \
                          strides=[1, 2, 2, 1], padding='SAME')
    #############
    #2nd Convolution
    conv = tf.nn.conv2d(pool, conv2_weights, \
                        strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], \
                          strides=[1, 2, 2, 1], padding='SAME')
    #############
    #1st fully connected
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], \
                                pool_shape[1] * pool_shape[2] * pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    #############
    #2nd fully connected
    fully_connected_2 = tf.matmul(hidden, fc2_weights) + fc2_biases
    #############
    #Return 
    #   1. The start node of the graph. (Used for input)
    #   2. final node in the graph. (Operation node for the 2)
    return input_placeholder, fully_connected_2


input_placeholder, logits = model()

# 2. load dataset to visualize embedding
data_sets = input_data.read_data_sets(test_path, validation_size=BATCH_SIZE)
batch_dataset, batch_labels = data_sets.validation.next_batch(BATCH_SIZE)

# 3. init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 4. load pre-trained model file
#saver = tf.train.Saver()
#saver.restore(sess, os.path.join(test_path, 'model.ckpt'))

feed_dict = {input_placeholder: batch_dataset.reshape([BATCH_SIZE, \
                                                       IMAGE_SIZE, \
                                                       IMAGE_SIZE, \
                                                       NUM_CHANNELS])}
activations = sess.run(logits, feed_dict)

assert((BATCH_SIZE, FULLY_CONNECTED_LAYER2_SIZE) == activations.shape)

# 5. summary embedding
embedder.summary_embedding(sess=sess, \
                           dataset=batch_dataset, \
                           embedding_list=[activations],
                           embedding_path=os.path.join(test_path, \
                                                       'embedding'),
                           image_size=IMAGE_SIZE, \
                           channel=NUM_CHANNELS, \
                           labels=batch_labels)

tf.reset_default_graph()
sess.close()

print("\n\tDONE: ", __file__)