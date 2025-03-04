import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. 数据准备：加载并预处理 MNIST 数据集
# ---------------------------
# 加载 MNIST 数据集，数据格式为 (num_samples, 28, 28)
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# 归一化数据到[-1, 1]（注意：生成器的最后一层用 tanh 激活函数输出范围为[-1,1]）
X_train = X_train.astype('float32') / 127.5 - 1.
# 扩展维度，使数据形状为 (num_samples, 28, 28, 1)
X_train = np.expand_dims(X_train, axis=-1)

# 定义 batch 大小
BUFFER_SIZE = X_train.shape[0]
BATCH_SIZE = 256

# 使用 tf.data 构建数据集，并打乱数据顺序
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ---------------------------
# 2. 构造生成器（Generator）
# ---------------------------
def build_generator():
    model = Sequential()
    # 输入噪声向量，维度设为100
    model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # 重塑为 7x7 的特征图，通道数为256
    model.add(Reshape((7, 7, 256)))
    
    # 第一个反卷积层，将尺寸放大至7x7（stride=1）
    model.add(Conv2DTranspose(128, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # 第二个反卷积层，放大尺寸至14x14（stride=2）
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # 第三个反卷积层，放大尺寸至28x28（stride=2），输出单通道图像，并用 tanh 激活函数输出[-1,1]范围
    model.add(Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh'))
    
    return model

# ---------------------------
# 3. 构造判别器（Discriminator）
# ---------------------------
def build_discriminator():
    model = Sequential()
    # 第一层卷积：提取特征，输出64个滤波器，尺寸缩小一半
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    # 第二层卷积：继续提取特征，输出128个滤波器
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    # 将多维特征展平，然后输出一个概率（使用 sigmoid 激活函数判断真假）
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# ---------------------------
# 4. 创建生成器和判别器实例，并编译判别器
# ---------------------------
generator = build_generator()
discriminator = build_discriminator()

# 定义判别器的优化器和损失函数
discriminator.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')

# ---------------------------
# 5. 构造 GAN 模型（将生成器和判别器级联）
# ---------------------------
# 在训练 GAN 时，我们希望更新生成器的权重，而固定判别器的参数，因此这里先冻结判别器
discriminator.trainable = False

# 构造 GAN 模型：输入噪声，经过生成器生成图像，然后判别器判断真假
gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)

# 编译 GAN 模型，目标是让生成器尽可能“欺骗”判别器，使得判别器认为生成的图像是真的
gan.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')

# ---------------------------
# 6. 训练 GAN 模型
# ---------------------------
# 定义训练参数
EPOCHS = 50      # 训练轮数
noise_dim = 100  # 噪声向量的维度
num_examples_to_generate = 16  # 用于生成样例图片的噪声数量

# 为了在训练过程中观察效果，固定一组随机噪声
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 创建保存生成图片的文件夹
if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

# 定义训练过程
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f'====== 开始 Epoch {epoch+1}/{epochs} ======')
        for image_batch in dataset:
            batch_size = image_batch.shape[0]
            
            # 1. 训练判别器
            # 生成噪声，生成假图片
            noise = tf.random.normal([batch_size, noise_dim])
            generated_images = generator(noise, training=True)
            
            # 拼接真实图片和生成的图片
            combined_images = tf.concat([image_batch, generated_images], axis=0)
            
            # 生成标签：真实图片为1，假图片为0
            labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
            # 为提高训练稳定性，给真实标签加上少量随机噪声
            labels += 0.05 * tf.random.uniform(labels.shape)
            
            # 训练判别器
            d_loss = discriminator.train_on_batch(combined_images, labels)
            
            # 2. 训练生成器
            # 生成噪声，目标是让判别器判断生成的图片为真实（标签为1）
            noise = tf.random.normal([batch_size, noise_dim])
            misleading_labels = tf.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, misleading_labels)
        
        # 每个 epoch 结束后输出当前的损失，并生成一些示例图片保存
        print(f'判别器损失: {d_loss:.4f}, 生成器损失: {g_loss:.4f}')
        generate_and_save_images(generator, epoch + 1, seed)

def generate_and_save_images(model, epoch, test_input):
    # 生成图像
    predictions = model(test_input, training=False)
    
    # 设定图像显示的网格尺寸
    fig = plt.figure(figsize=(4,4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # 将像素值从 [-1,1] 转换回 [0,1]
        plt.imshow((predictions[i, :, :, 0] + 1) / 2, cmap='gray')
        plt.axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    # 保存生成的图片
    plt.savefig(f'generated_images/image_at_epoch_{epoch:04d}.png')
    plt.close(fig)

# 开始训练
train(train_dataset, EPOCHS)
