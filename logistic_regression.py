import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01
training_steps = 100
batch_size = 100
display_steps = 1

# training data
mnist = input_data.read_data_sets("/tmp/mnist_data/", one_hot=True)

# tf input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# model
# weight
W = tf.Variable(tf.zeros([784, 10]), name='weight')
b = tf.Variable(tf.zeros([10]), name='bias')

# linear model
predict = tf.nn.softmax(tf.matmul(x, W) + b)

# minimize squares error
cost = -tf.reduce_sum(y*tf.log(predict))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# init variables
init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # fit all train data
    for epoch in range(training_steps):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([train, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c/total_batch
        
        # display every 50 steps
        if epoch%display_steps == 0:
            print('Epoch: %04d' % (epoch+1), \
                'cost = {:.9f}'.format(avg_cost))

    print('Optimize finished!')

    correct_count = tf.equal(tf.argmax(predict, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_count, tf.float32))
    
    print('acc = {:.9f}'.format(sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
    # equivalent way:
    # print('acc = {:.9f}'.format(acc.eval({x: mnist.test.images, y: mnist.test.labels})))