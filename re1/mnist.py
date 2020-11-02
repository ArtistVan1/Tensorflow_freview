import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(train_data,train_label), (test_data,test_label) = mnist.load_data()
print(train_data.shape)
print(test_data.shape)
print(train_label[0]) #5
train_data = train_data /255.0
test_data = test_data / 255.0

# class MyMode(tf.keras.Model):
#     def __init__(self):
#         super(MyMode,self).__init__()
#         self.d1 = tf.keras.layers.Flatten(),
#         self.d2 = tf.keras.layers.Dense(128,activation='relu'),
#         self.d3 = tf.keras.layers.Dense(10,activation='softmax')
#
#     def call(self, x):
#         x1 = self.d1(x)
#         x2 = self.d2(x1)
#         y = self.d3(x2)
#         return y
#
# model = MyMode()
class MyMode(tf.keras.Model):
    def __init__(self):
        super(MyMode,self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        self.d2 = tf.keras.layers.Dense(10,activation='softmax')

    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128,activation='relu'),
#     tf.keras.layers.Dense(10,activation='softmax')
# ])
model = MyMode()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_data,train_label,batch_size=32,epochs=5,validation_data=(test_data,test_label),validation_freq=1)
model.summary()