import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[0:-30]
y_train = y_data[0:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train,dtype=tf.float32)
x_test = tf.cast(x_test,dtype=tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

epoch = 500
lr = 0.1
test_acc = []
loss = 0
train_loss = []
loss_all = 0

for step in range(0,epoch): #epoch
    for index,(x_train,y_train) in enumerate(train_db): #batch
        with tf.GradientTape() as tape:
            y_ = tf.matmul(x_train,w1)+b1
            y_ = tf.nn.softmax(y_)
            y = tf.one_hot(y_train,depth=3)
            loss = tf.reduce_mean(tf.square(y-y_))
            loss_all +=loss.numpy()  #画图用一般的数值类型就可
        grad = tape.gradient(loss,[w1,b1])
        w1.assign_sub(lr*grad[0])
        b1.assign_sub(lr*grad[1])

    print("Epoch:{}, Loss:{}".format(step,loss_all/4))
    train_loss.append(loss_all/4)
    loss_all = 0

    total_correct_num, total_num = 0,0
    for index,(x_test,y_test) in enumerate(test_db):
        y = tf.matmul(x_test,w1)+b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y,axis=1)
        #print(pred.dtype)
        pred = tf.cast(pred,dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct_num += int(correct)
        total_num +=x_test.shape[0]

    acc = total_correct_num / total_num
    test_acc.append(acc)
    print("Test_acc", acc)
    print("-----------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()















