import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math

class ConvBlock(tf.keras.layers.Layer):
	def __init__(self, kernel_num, kernel_size, strides, padding='same'):
		super(ConvBlock, self).__init__()
		# conv layer
		self.conv = tf.keras.layers.Conv2D(kernel_num, 
						kernel_size=kernel_size, 
						strides=strides, padding=padding)

		# batch norm layer
		self.bn   = tf.keras.layers.BatchNormalization()


	def call(self, input_tensor, training=False):
		x = self.conv(input_tensor)
		x = self.bn(x, training=training)
		x = tf.nn.relu(x)
		
		return x

class DeConvBlock(tf.keras.layers.Layer):
	def __init__(self, kernel_num, kernel_size, padding='same'):
		super(DeConvBlock, self).__init__()
		# conv layer
		self.deconv = tf.keras.layers.Conv2DTranspose(filters=kernel_num, kernel_size=kernel_size, strides=(2, 2), padding="SAME")

		# batch norm layer
		self.bn   = tf.keras.layers.BatchNormalization()


	def call(self, input_tensor, training=False):
		x = self.deconv(input_tensor)
		x = tf.nn.relu(x)
		x = self.bn(x, training=training)
		
		return x

class SEBlock(tf.keras.layers.Layer):
	def __init__(self, channels):
		super(SEBlock, self).__init__()
		self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
		self.excitation1 = tf.keras.layers.Dense(channels//16)
		self.excitation2 = tf.keras.layers.Dense(channels)
		self.reshape = tf.keras.layers.Reshape((1,1,channels))


	def call(self, input_tensor):
		residual = input_tensor
		x = self.squeeze(input_tensor)
		x = self.excitation1(x)
		x = tf.nn.relu(x)
		x = self.excitation2(x)
		x = tf.nn.sigmoid(x)
		x = self.reshape(x)
		
		return tf.multiply(x, residual)

class ResBlock(tf.keras.layers.Layer):
	def __init__(self, channels, padding='same'):
		super(ResBlock, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=(3,3), strides=(1,1), padding=padding)
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size=(3,3), strides=(1,1), padding=padding)
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.scale = SEBlock(channels)


	def call(self, input_tensor, training=False):
		residual = input_tensor
		x = self.conv1(input_tensor)
		x = tf.nn.relu(x)
		x = self.bn1(x, training=training)
		x = self.conv2(x)
		x = tf.nn.relu(x)
		x = self.bn2(x, training=training)
		x = self.scale(x)
		x = tf.keras.layers.add([x, residual])

		return x


class StressNet(tf.keras.Model):
	def __init__(self):
		super(StressNet, self).__init__()

		# the first conv module
		self.conv1 = ConvBlock(16, (3,3), (1,1))
		self.conv2 = ConvBlock(32, (3,3), (2,2))
		self.conv3 = ConvBlock(64, (3,3), (2,2))
		self.conv4 = ConvBlock(64, (3,3), (2,2))
		self.conv5 = ConvBlock(128, (3,3), (2,2))

		self.res1 = ResBlock(128)
		self.res2 = ResBlock(128)
		self.res3 = ResBlock(128)
		self.res4 = ResBlock(128)

		self.deconv1 = DeConvBlock(64, (3,3))
		self.deconv2 = DeConvBlock(64, (3,3))
		self.deconv3 = DeConvBlock(32, (3,3))
		self.deconv4 = DeConvBlock(16, (3,3))
		self.outConv = tf.keras.layers.Conv2D(1, kernel_size=(3,3), strides=(1,1), padding='same')


	def call(self, input_tensor, training=False, **kwargs):
		
		# forward pass 
		x = self.conv1(input_tensor, training=training)
		x = self.conv2(x, training=training)
		x = self.conv3(x, training=training)
		x = self.conv4(x, training=training)
		x = self.conv5(x, training=training)

		x = self.res1(x, training=training)
		x = self.res2(x, training=training)
		x = self.res3(x, training=training)
		x = self.res4(x, training=training)

		x = self.deconv1(x, training=training)
		x = self.deconv2(x, training=training)
		x = self.deconv3(x, training=training)
		x = self.deconv4(x, training=training)
		x = self.outConv(x)

		return x

	def build_graph(self, shape):
		x = tf.keras.layers.Input(shape=shape)
		return tf.keras.Model(inputs=[x], outputs=self.call(x))


def main():
	input_dim = 128
	output_dim = 128
	norm_fac = 1e-5
	total_train = 60
	test_start_ind = 0
	batch_size = 60
	num_batches = total_train//batch_size
	epochs = 1000
	learn_rate = 5e-4
	is_train = False
	if is_train:
		test_end_ind = 10
	else:
		test_end_ind = 100
	savefileid = 'save/saveprop'+str(total_train)

	np.random.seed(0)

	dens_data_pre1 = np.load('dataset/cantilever/dens128.npy')
	dens_data = dens_data_pre1[:total_train]
	print('dens_data shape:',dens_data.shape)
	load_data_pre1 = np.load('dataset/cantilever/load128.npy')
	load_data = load_data_pre1[:total_train]
	x_data = np.concatenate((dens_data.reshape(-1,input_dim,input_dim,1),load_data.reshape(-1,input_dim,input_dim,1)),axis=3)
	comp_data_pre1 = np.load('dataset/cantilever/comp128.npy')
	y_data = comp_data_pre1[:total_train]
	test_data1 = np.load('dataset/cantilever/dens128test.npy')
	test_data2 = np.load('dataset/cantilever/load128test.npy')
	test_data3 = np.load('dataset/cantilever/comp128test.npy')

	perm = np.random.permutation(x_data.shape[0])
	shuffle_input = x_data[perm]
	shuffle_output = 1/norm_fac*y_data[perm].reshape(-1,output_dim,output_dim,1)

	x_train = tf.convert_to_tensor(shuffle_input[0:total_train],dtype=tf.float32)
	y_train = tf.convert_to_tensor(shuffle_output[0:total_train],dtype=tf.float32)
	print('x_train shape = {}, y_train shape = {}'.format(x_train.shape,y_train.shape))

	perm_test = np.random.permutation(test_data1.shape[0])
	x_test_pre = np.concatenate((test_data1[perm_test].reshape(-1,input_dim,input_dim,1),test_data2[perm_test].reshape(-1,input_dim,input_dim,1)),axis=3)
	y_test_pre = 1/norm_fac*test_data3[perm_test].reshape(-1,output_dim,output_dim,1)
	x_test = tf.convert_to_tensor(x_test_pre[test_start_ind:test_end_ind],dtype=tf.float32)
	y_test = tf.convert_to_tensor(y_test_pre[test_start_ind:test_end_ind],dtype=tf.float32)
	print('x_test shape = {}, y_test shape = {}'.format(x_test.shape,y_test.shape))

	print("x train shape: {0}, y train shape: {1}, x test shape: {2}, y test shape: {3}".format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))
	print("x train max min: {0} {1}, y train max min: {2} {3}, x test max min: {4} {5}, y test max min: {6} {7}".format(np.amax(x_train),np.amin(x_train),np.amax(y_train),np.amin(y_train),np.amax(x_test),np.amin(x_test),np.amax(y_test),np.amin(y_test)))

	# init model object
	if is_train == False:
		model = tf.keras.models.load_model(savefileid, compile=False)
	else:
		model = StressNet()

	# Instantiate an optimizer to train the model.
	optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
	# Instantiate a loss function.
	loss_fn = tf.keras.losses.MeanSquaredError()

	# Prepare the metrics.
	train_acc_metric = tf.keras.metrics.CosineSimilarity(axis=1)
	val_acc_metric   = tf.keras.metrics.CosineSimilarity(axis=1)

	if is_train == True:
		# #tensorboard writer 
		train_writer = tf.summary.create_file_writer('logs/train/')
		test_writer  = tf.summary.create_file_writer('logs/test/')

		@tf.function
		def train_step(x, y):
			with tf.GradientTape() as tape:
				prediction = model(x, training=False)
				train_loss_value = loss_fn(y, prediction)
			grads = tape.gradient(train_loss_value, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			train_acc_metric.update_state(tf.math.abs(y), tf.math.abs(prediction))
			
			return train_loss_value

		@tf.function
		def test_step(x, y):
			val_pred = model(x, training=False)
			# Compute the loss value f
			val_loss_value = loss_fn(y, val_pred)
			# Update val metrics
			val_acc_metric.update_state(tf.math.abs(y), tf.math.abs(val_pred))

			return val_loss_value
			
		# custom training loop 
		train_loss_curve = [] 
		test_loss_curve = [] 
		for epoch in range(epochs):
			# batch training 
			# Iterate over the batches of the dataset.
			time1=time.time()
			for train_batch_step in range(num_batches):
				train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.int64)
				x_batch_train = x_train[train_batch_step*batch_size:(train_batch_step+1)*batch_size]
				y_batch_train = y_train[train_batch_step*batch_size:(train_batch_step+1)*batch_size]
				train_loss_value = train_step(x_batch_train, y_batch_train)
				train_loss_curve.append(train_loss_value)

			# write training loss and accuracy to the tensorboard
			with train_writer.as_default():
				tf.summary.scalar('loss', train_loss_value, step=epoch)
				tf.summary.scalar('accuracy', train_acc_metric.result(), step=epoch) 

			# evaluation on validation set 
			# Run a validation loop at the end of each epoch.
			val_loss_value = test_step(x_test, y_test)
			test_loss_curve.append(val_loss_value)

			# write test loss and accuracy to the tensorboard
			with test_writer.as_default():
				tf.summary.scalar('val loss', val_loss_value, step=epoch)
				tf.summary.scalar('val accuracy', val_acc_metric.result(), step=epoch) 

			template = 'epoch: {} loss: {}  acc: {} val loss: {} val acc: {} time: {}'
			if (epoch+1)%100 == 0:
				print(template.format(
					epoch + 1,
					train_loss_value, float(train_acc_metric.result()),
					val_loss_value, float(val_acc_metric.result()),
					time.time() - time1
				))
			
			# Reset metrics at the end of each epoch
			if (epoch+1) < epochs:
				train_acc_metric.reset_states()
				val_acc_metric.reset_states()

		model.save(savefileid)
		print('after model saved!')
	prediction = model.predict(x_test)

	error = np.mean(np.square(prediction-y_test))
	val_acc_metric.update_state(tf.math.abs(y_test), tf.math.abs(prediction))
	acc = val_acc_metric.result()

	print("mean square error is {0}, accuracy is {1}".format(error,math.acos(acc)))

	if is_train==False:
		fig = plt.figure()
		for i in range(8):
			ax = fig.add_subplot(2,8,i+1)
			ax.imshow(np.fliplr(np.log(norm_fac*y_test[i+4]+1e-6).reshape(output_dim,output_dim).T), cmap='gray')
			ax.set_xticks([]) 
			ax.set_yticks([]) 
			ax = fig.add_subplot(2,8,i+9)
			ax.imshow(np.fliplr(np.log(np.maximum(norm_fac*prediction[i+4],0)+1e-6).reshape(output_dim,output_dim).T), cmap='gray')
			ax.set_xticks([]) 
			ax.set_yticks([]) 
		plt.show()

if __name__=='__main__':
	main()
