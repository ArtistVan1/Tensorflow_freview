import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)

d = tf.random.normal([2,2],mean=1,stddev=2)
print(d)
e = tf.random.truncated_normal([3,3],mean=1.5,stddev=2.5)
print(e)
d = tf.random.uniform([2,2],minval=0.5,maxval=3)
print(d)

with tf.Session():
    print(d.eval())