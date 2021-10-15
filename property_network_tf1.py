
import tensorflow as tf
import numpy as np
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



def FirstOrder():
    x = tf.placeholder(tf.float32, [None, input_dim, input_dim, 2])  
    y = tf.placeholder(tf.float32, [None, output_dim, output_dim, 1])

    prediction = network(x)
    print('--network built--')
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(y-prediction)))

    train = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    iii = 0
    best_loss = 100000

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        savefileid = 'save_prop/model'+str(TotTrainData)
        for step in range(num_iters):
            for batch_i in range(num_batch):
                x_batch = x_data[batch_i*batch_size:(batch_i+1)*batch_size,:]
                y_batch = y_data[batch_i*batch_size:(batch_i+1)*batch_size,:]
                sess.run([train],feed_dict={x:x_batch,y:y_batch})
                losses.append(np.mean(sess.run(loss,feed_dict={x:x_batch,y:y_batch})))
                iii = iii+1
            print('iter: {0}, loss = {1}'.format(step,losses[iii-1]))
            if (step+1)%10 == 0 :
                if losses[-1] < best_loss:
                    saver.save(sess, savefileid)
                    best_loss = losses[-1]


losses = []
learn_rate = 1e-4
input_dim = 128
output_dim = 128
norm_fac = 1e-4#5e-5
num_iters = 1000
TotTrainData = 100
TotTestData = 100
batch_size = 100
num_batch = (TotTrainData//batch_size)

d1_0 = np.load('dataset/density_arr.npy')
d2_0 = np.load('dataset/load_arr.npy')
d3_0 = np.load('dataset/compliance_arr.npy')

perm = np.random.permutation(d1_0.shape[0])
d1 = d1_0[perm]
d2 = d2_0[perm]
d3 = d3_0[perm]

dataset1 = d1
dataset2 = d2
dataset3 = d3/norm_fac

teststart = 100
testend = 200

x_data1 = dataset1[0:TotTrainData,:].reshape(TotTrainData,input_dim,input_dim,1)
x_data2 = dataset2[0:TotTrainData,:].reshape(TotTrainData,input_dim,input_dim,1)
x_data = np.concatenate((x_data1,x_data2),axis=3)
y_data = dataset3[0:TotTrainData,:].reshape(TotTrainData,output_dim,output_dim,1)
print('train data shape:',x_data.shape, y_data.shape)
x_data_test1 = dataset1[teststart:testend,:].reshape(TotTestData,input_dim,input_dim,1)
x_data_test2 = dataset2[teststart:testend,:].reshape(TotTestData,input_dim,input_dim,1)
x_data_test = np.concatenate((x_data_test1,x_data_test2),axis=3)
y_data_test = dataset3[teststart:testend,:].reshape(TotTestData,output_dim,output_dim,1)
print('test data shape:',x_data_test.shape, y_data_test.shape)

print('x data max min:',np.amax(x_data),np.amin(x_data))
print('y data max min:',np.amax(y_data),np.amin(y_data))
print('x data test max min:',np.amax(x_data_test),np.amin(x_data_test))
print('y data test max min:',np.amax(y_data_test),np.amin(y_data_test))
print('--load finish--')
FirstOrder()

