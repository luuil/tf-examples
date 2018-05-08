import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(2)
b = tf.constant(3)
print('2 + 3 = %i'% sess.run(a+b))

c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)
add = tf.add(c, d)
mul = tf.multiply(c, d)
add_feed = {c: 2, d: 3}
mul_feed = {c: 2, d: 3}
print('add: %i' % sess.run(add, feed_dict=add_feed))
print('mul: %i' % sess.run(mul, feed_dict=mul_feed))

m1 = tf.constant([[3., 3.]])
m2 = tf.constant([[2.], [2.]])
mmul = tf.matmul(m1, m2)
print('mat mul: %i' % sess.run(mmul))