import tensorflow as tf
tf.enable_eager_execution()
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print(element)

print("/////")
features = tf.constant([1,2,3,4],dtype=tf.int64)
labels = tf.constant([0,0,1,1])
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
for each in dataset:
    print(each)