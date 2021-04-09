
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import conv2d_transpose
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import fully_connected as dense


def get_loss(loss):
    print(loss)
    losses.append(loss)

def max_pool_2x2(x):
    return(tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

def Global_Average_Pooling(x):
    return global_avg_pool(x)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.nn.tanh(x)

def bn(x):
    return tf.layers.batch_normalization(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def dropout(x, keep_prob=0.5):
    return tf.nn.dropout(x, keep_prob)

def seblock(x, in_cn):

    squeeze = Global_Average_Pooling(x)

    with tf.variable_scope('sq'):
        h = dense(squeeze, in_cn//16, activation_fn=None)
        excitation = relu(h) 
 
    with tf.variable_scope('ex'):
        h = dense(excitation, in_cn, activation_fn=None)
        excitation = sigmoid(h) 
        excitation = tf.reshape(excitation, [-1, 1, 1, in_cn]) 

    return x * excitation

def residual_block(x, cn, scope_name):
    with tf.variable_scope(scope_name):
        shortcut = x
        x1 = bn(relu(conv2d(x, cn, kernel_size=(5, 5), stride=(1, 1), padding='SAME', activation_fn=None)))
        x2 = bn(relu(conv2d(x1, cn, kernel_size=(5, 5), stride=(1, 1), padding='SAME', activation_fn=None)))

        x3 = seblock(x2, cn) 

    return x3 + shortcut

def network(gen_inputs):

    with tf.variable_scope('encoder'):

        x1 = relu(bn(conv2d(gen_inputs, 16, kernel_size=(5, 5), stride=(1, 1), padding='SAME', activation_fn=None)))
        x2 = relu(bn(conv2d(x1, 32, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))
        x3 = relu(bn(conv2d(x2, 64, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))
        x4 = relu(bn(conv2d(x3, 64, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))
        x5 = relu(bn(conv2d(x4, 128, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))

    with tf.variable_scope('residual'):

        x6 = residual_block(x5, 128, 'res1')
        x7 = residual_block(x6, 128, 'res2')
        x8 = residual_block(x7, 128, 'res3')
        x9 = residual_block(x8, 128, 'res4')
        x10 = residual_block(x9, 128, 'res5')

    with tf.variable_scope('decoder'):

        x11 = bn(relu(conv2d_transpose(x10, 64, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))
        x12 = bn(relu(conv2d_transpose(x11, 64, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))
        x13 = bn(relu(conv2d_transpose(x12, 32, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))
        x14 = bn(relu(conv2d_transpose(x13, 16, kernel_size=(5, 5), stride=(2, 2), padding='SAME', activation_fn=None)))
        net = conv2d(x14, 1, kernel_size=(5, 5), stride=(1, 1), padding='SAME', activation_fn=None)


    return net
