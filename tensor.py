import tensorflow as tf
a = tf.constant([1,5],dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape)

import numpy as np
a=np.arange(0,5)
b = tf.convert_to_tensor(a,dtype=tf.int64)
print(a)
print(b)