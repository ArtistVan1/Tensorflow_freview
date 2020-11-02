import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print(len(train_labels)) #60000
print(train_images.shape)
print(test_images.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #该层没有要学习的参数
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10)
]
)

model.compile(optimizer='adam',
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10)

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
print(np.argmax(predictions[0]))