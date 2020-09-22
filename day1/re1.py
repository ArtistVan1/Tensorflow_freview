import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

print(y_data)

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(y_data)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[0:-30]
y_train = y_data[0:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)

w1 = tf.Variable(tf.random.truncated_normal([4,3],mean=0.1,stddev=0.1))
b1 = tf.Variable(tf.random.truncated_normal([3],mean=0.1,stddev=0.1))

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

epochs = 500
lr = 0.1
train_loss = []
loss_all=0
test_acc = []
for step in range(epochs):
    for index,(train,label) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y_ = tf.matmul(train,w1) + b1
            y_ = tf.nn.softmax(y_)
            label = tf.one_hot(label,depth=3)
            loss = tf.reduce_mean(tf.square(y_-label))
            loss_all +=loss.numpy()
        grad = tape.gradient(loss,[w1,b1])

        w1.assign_sub(grad[0]*lr)
        b1.assign_sub(grad[1]*lr)
    print("Epoch:{}, Loss:{}".format(step,loss_all/4))
    train_loss.append(loss_all/4)
    loss_all = 0

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

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








