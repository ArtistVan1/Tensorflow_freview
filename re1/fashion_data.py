import tensorflow as tf
import numpy as np

fashion = tf.keras.datasets.fashion_mnist
(train_data,train_label),(test_data,test_label) = fashion.load_data()

print(train_data.shape)
print(test_data.shape)
print(train_label[0])
train_data = train_data / 255.0
test_data = test_data / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_data,train_label,batch_size=32,epochs=5,validation_data=(test_data,test_label),validation_freq=1)

model.summary()