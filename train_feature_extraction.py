import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import time

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
	train = pickle.load(f)
class_num = max(y_train) + 1

X_train, y_train = train['features'], train['labels']

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)
# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
x = tf.image.resize_image(x, (227,227))
y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, class_num)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (tf.get_shape(fc7).as_list(-1), class_num)
mu = 0
sigma = 0.1
learning_rate = 0.0001
EPOCHS = 50
BATCH_SIZE = 128

wfc8 = tf.Variable(tf.truncated_normal(shape = shape, mean = mu, stddev = sigma))
bfc8 = tf.Variable(tf.zeros(class_num))
fc8 = tf.matmul(fc7, wfc8) + bfc8
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=fc8)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_oper = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_one_hot,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# TODO: Train and evaluate the feature extraction model.
def evaluate(x_data, y_data):
	num_example = len(x_data)
	total_accu = 0
	sess = tf.get_default_session()
	for i in range(0, num_example, BATCH_SIZE):
		batch_x, batch_y = x_data[i:i+BATCH_SIZE], y_data[i:i+BATCH_SIZE]
		accuracy = sess.run(accuracy_operation, feed_dict={x:x_data, y:y_data})
		total_accu += accuracy * len(batch_x)

	return total_accu/num_example

saver = tf.train.Saver()

# Train 
with tf.Session() as sess:
	sess.run(tf.global_variable_initializer())
	for epoch in EPOCHS:
		time1 = time.time()
		for i in range(0, X_train, BATCH_SIZE):
			x_data, y_data = X_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]
			sess.run(train_oper, feed_dict={x_data, y:y_data})
		time2 = time.time()
		train_acc = evaluate(X_train, y_train)
		valid_acc = evaluate(X_valid, y_valid)

		if epoch%10 == 0:
			print('EPOCH: {}'.format(epoch +1), 'Train Accuracy: ', train_acc, 'Validation Accuracy: ', valid_acc, 'time: ', time2-time1)

	saver.save('./trained_model')
	print('model saved.')

# Test
# with tf.Session() as sess:
# 	saver.restore(sess, tf.train.latest_checkpoint('.'))
# 	prediction = sess.run(tf.argmax(logits, 1), feed_dict={x:x})










