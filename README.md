# -Fake-Image-Detection-using-GAN-Models
# Common
import os
import keras
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf

# Data
from tensorflow.image import resize
from keras.preprocessing.image import load_img, img_to_array

# Data Viz
import matplotlib.pyplot as plt

# Model
from keras import Sequential
from keras.initializers import RandomNormal
from keras.layers import Conv2D
from keras.layers import Layer
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.layers import add
from keras.layers import multiply
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D

# Model Functions
from keras.models import Model, load_model

# Optimizer
from tensorflow.keras.optimizers import Adam


# Model Viz
from tensorflow.keras.utils import plot_model


Data
def load_image(path, SIZE):
    '''Take in the path to the image and load + resize it'''
    img = tf.cast(resize(img_to_array(load_img(path))/255., (SIZE,SIZE)), tf.float32)
    return img
    
def load_images(paths, SIZE=256):
    '''Takes in the list of all Image Paths and load them one by one'''
    images = np.zeros(shape=(len(paths), SIZE, SIZE, 3))
    for i, path in tqdm(enumerate(paths), desc="Loading"):
        img = load_image(path, SIZE)
        images[i] = img
    return images

def show_image(image, title=None):
    '''Takes in an Image and plot it with the help of Matplotlib'''
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
# Image Paths
image_paths = sorted(glob("../input/celeba-dataset/img_align_celeba/img_align_celeba" + "/*.jpg"))
print(f"Total Number of Images : {len(image_paths)}")
Total Number of Images : 202599

images = load_images(image_paths[:1000])
Loading: 0it [00:00, ?it/s]2022-09-20 23:42:12.554453: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:12.667558: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:12.668326: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:12.671023: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-09-20 23:42:12.671331: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:12.672038: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:12.672673: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:15.062246: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:15.063350: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:15.064338: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-09-20 23:42:15.065254: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 15401 MB memory:  -> device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0, compute capability: 6.0
Loading: 1000it [00:12, 80.78it/s]
Let's see how tough it could be for the model to Generate Human Faces by visualizing these Faces.

Data Visualization
plt.figure(figsize=(15,8))
for i in range(1,11):
    id = np.random.randint(len(images))
    image = images[id]
    plt.subplot(2,5,i)
    show_image(image)    
plt.show()

Encoder
The Generator of a Pix2Pix GAN is based on a UNet Autoencoder. For that we need 2 networks, the encoder and decoder network.

The Encoder consist of a convolutional 2D layer with strides 2, so it will downsample it's inputs. Followed by a BatchNormalization layer and a LeakyReLU Activaton.

class Encoder(Layer):
    
    def __init__(self, filters, apply_bn=True, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
        self.filters = filters
        self.apply_bn = apply_bn
        
        self.c = Conv2D(filters, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)
        self.bn = BatchNormalization()
        self.act = LeakyReLU()
    
    def call(self, X):
        X = self.c(X)
        if self.apply_bn:
            X = self.bn(X)
        X = self.act(X)
        return X
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "apply_bn":self.apply_bn
        }

        Decoder
The Decoder is just the opposite of the encoder. It uses a convolutional transpose layer to upsample its inputs and a ReLU activation function.

class Decoder(Layer):
    
    def __init__(self, filters, apply_dropout=True, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        
        self.filters = filters
        self.apply_drop = apply_dropout
        
        self.c = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)
        self.bn = BatchNormalization()
        self.drop = Dropout(0.5)
        self.act = LeakyReLU()
    
    def call(self, X):
        X = self.c(X)
        X = self.bn(X)
        if self.apply_drop:
            X = self.drop(X)
        X = self.act(X)
        return X
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "apply_drop":self.apply_drop
        }
I will be trying this out, it should just fit into the Memory.

Generator
We have both of our networks i.e Encoder and Decoder ready. It's time to combine them and build a UNet Generator.

# Input and Output Layer
SIZE = 256
init = RandomNormal(stddev=0.02)
inputs = Input(shape=(SIZE, SIZE, 3), name="InputLayer")
last = Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='tanh', kernel_initializer=init, name="OutputLayer")


# Encoder Network
e1 = Encoder(64, apply_bn=False, name="Encoder1")(inputs)
e2 = Encoder(128, name="Encoder2")(e1)
e3 = Encoder(256, name="Encoder3")(e2)
e4 = Encoder(512, name="Encoder4")(e3)
e5 = Encoder(512, name="Encoder5")(e4)
e6 = Encoder(512, name="Encoder6")(e5)

# Encoding Layer
e7 = Encoder(512, name="Encoder7")(e6)

# Decoder Network
d1 = Decoder(512, apply_dropout=True, name="Decoder1")(e7)
c1 = concatenate([d1, e6])  # We will try Adding Attention Gate at this Stage

d2 = Decoder(512, apply_dropout=True, name="Decoder2")(c1)
c2 = concatenate([d2, e5])

d3 = Decoder(512, apply_dropout=True, name="Decoder3")(c2)
c3 = concatenate([d3, e4])

d4 = Decoder(512, name="Decoder4")(c3)
c4 = concatenate([d4, e3])


d5 = Decoder(256, name="Decoder5")(c4)
c5 = concatenate([d5, e2])

d6 = Decoder(128, name="Decoder6")(c5)
c6 = concatenate([d6, e1])

d7 = Decoder(64, name="Decoder7")(c6)

out = last(d7)

generator = Model(
    inputs=inputs,
    outputs=out,
    name="Generator"
)

So this is our generator. It will be easy to understand once we visualize it.

generator.summary()
Model: "Generator"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
InputLayer (InputLayer)         [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
Encoder1 (Encoder)              (None, 128, 128, 64) 1728        InputLayer[0][0]                 
__________________________________________________________________________________________________
Encoder2 (Encoder)              (None, 64, 64, 128)  74240       Encoder1[0][0]                   
__________________________________________________________________________________________________
Encoder3 (Encoder)              (None, 32, 32, 256)  295936      Encoder2[0][0]                   
__________________________________________________________________________________________________
Encoder4 (Encoder)              (None, 16, 16, 512)  1181696     Encoder3[0][0]                   
__________________________________________________________________________________________________
Encoder5 (Encoder)              (None, 8, 8, 512)    2361344     Encoder4[0][0]                   
__________________________________________________________________________________________________
Encoder6 (Encoder)              (None, 4, 4, 512)    2361344     Encoder5[0][0]                   
__________________________________________________________________________________________________
Encoder7 (Encoder)              (None, 2, 2, 512)    2361344     Encoder6[0][0]                   
__________________________________________________________________________________________________
Decoder1 (Decoder)              (None, 4, 4, 512)    2361344     Encoder7[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 4, 4, 1024)   0           Decoder1[0][0]                   
                                                                 Encoder6[0][0]                   
__________________________________________________________________________________________________
Decoder2 (Decoder)              (None, 8, 8, 512)    4720640     concatenate[0][0]                
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 8, 8, 1024)   0           Decoder2[0][0]                   
                                                                 Encoder5[0][0]                   
__________________________________________________________________________________________________
Decoder3 (Decoder)              (None, 16, 16, 512)  4720640     concatenate_1[0][0]              
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 16, 16, 1024) 0           Decoder3[0][0]                   
                                                                 Encoder4[0][0]                   
__________________________________________________________________________________________________
Decoder4 (Decoder)              (None, 32, 32, 512)  4720640     concatenate_2[0][0]              
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 32, 32, 768)  0           Decoder4[0][0]                   
                                                                 Encoder3[0][0]                   
__________________________________________________________________________________________________
Decoder5 (Decoder)              (None, 64, 64, 256)  1770496     concatenate_3[0][0]              
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 64, 64, 384)  0           Decoder5[0][0]                   
                                                                 Encoder2[0][0]                   
__________________________________________________________________________________________________
Decoder6 (Decoder)              (None, 128, 128, 128 442880      concatenate_4[0][0]              
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 128, 128, 192 0           Decoder6[0][0]                   
                                                                 Encoder1[0][0]                   
__________________________________________________________________________________________________
Decoder7 (Decoder)              (None, 256, 256, 64) 110848      concatenate_5[0][0]              
__________________________________________________________________________________________________
OutputLayer (Conv2DTranspose)   (None, 256, 256, 3)  1731        Decoder7[0][0]                   
==================================================================================================
Total params: 27,486,851
Trainable params: 27,476,995
Non-trainable params: 9,856
__________________________________________________________________________________________________
Generator Visualization
plot_model(generator, "Generator.png", show_shapes=True)

Discriminator
As Generator is ready we need a Discriminator so that it can discriminate the generated images and the real images.

class ZeroPadBlock(Layer):
    
    def __init__(self, **kwargs):
        super(ZeroPadBlock, self).__init__(**kwargs)
        
        init = RandomNormal(stddev=0.02)
        self.z1 = ZeroPadding2D()
        self.cT = Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', kernel_initializer=init, use_bias=False)
        self.bn = BatchNormalization()
        self.act = LeakyReLU()
        self.z2 = ZeroPadding2D()
    def call(self, X):
        X = self.z1(X)
        X = self.cT(X)
        X = self.bn(X)
        X = self.act(X)
        X = self.z2(X)
        return X
        
init = RandomNormal(stddev=0.02)
gen_input = Input(shape=(SIZE, SIZE, 3), name="GeneratedImage")

x = Encoder(64, apply_bn=False, name="DisBlock1")(gen_input)
x = Encoder(128, name="DisBlock2")(x)
x = Encoder(256, name="DisBlock3")(x)
x = Encoder(512, name="DisBlock4")(x)
x = ZeroPadBlock()(x)
x = Conv2D(1, kernel_size=4, strides=1, padding='valid', kernel_initializer=init)(x)
    
discriminator = Model(
    inputs=gen_input,
    outputs=[x],
    name="Discriminator"
)
discriminator.summary()
Model: "Discriminator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
GeneratedImage (InputLayer)  [(None, 256, 256, 3)]     0         
_________________________________________________________________
DisBlock1 (Encoder)          (None, 128, 128, 64)      1728      
_________________________________________________________________
DisBlock2 (Encoder)          (None, 64, 64, 128)       74240     
_________________________________________________________________
DisBlock3 (Encoder)          (None, 32, 32, 256)       295936    
_________________________________________________________________
DisBlock4 (Encoder)          (None, 16, 16, 512)       1181696   
_________________________________________________________________
zero_pad_block (ZeroPadBlock (None, 23, 23, 512)       4196352   
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 20, 20, 1)         8193      
=================================================================
Total params: 5,758,145
Trainable params: 5,755,329
Non-trainable params: 2,816
_________________________________________________________________
Discriminator Visualization
plot_model(discriminator, "Discriminator.png", show_shapes=True)

Functions
# Params
PATCH_SIZE = 20
BATCH_SIZE = 8

# Data
dataset = images
train_ds = tf.data.Dataset.from_tensor_slices(dataset).batch(BATCH_SIZE, drop_remainder=True).prefetch(1)
2022-09-20 23:42:27.849828: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1572864000 exceeds 10% of free system memory.
2022-09-20 23:42:29.567121: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1572864000 exceeds 10% of free system memory.
Creating a function for generating the real and fake labels.

def generate_labels():
    real_labels = tf.ones(shape=(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3))
    fake_labels = tf.zeros(shape=(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3))
    return real_labels, fake_labels
Pix2Pix GAN
First of all, we need to compile the Discriminator and the Generator.

# Compile Discriminator
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
    metrics=['accuracy']
)

# Set Non-Trainable
discriminator.trainable = False
# GAN Layers
input_img = Input(shape=(256,256,3), name="InputImage")
gen = generator(input_img)
dis = discriminator(gen)

# GAN
gan = Model(
    inputs=input_img,
    outputs=[gen, dis],
    name="Pix2PixGAN"
)


# Compiling
gan.compile(
    loss=['mae', 'binary_crossentropy'],
    optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
    loss_weights=[100,1]
)
The Pix2Pix GAN is ready. It's time to train the model.

Training Functions
def show_predictions(n_images=5):
    for i in range(n_images):
        plt.figure(figsize=(10,8))
        noise = tf.random.normal(shape=(1,256,256,3))
        gen_face = generator.predict(noise)[0]
        show_image(gen_face, title="Generated Face")
        plt.show()
Let's have a look at the Generator before training.

show_predictions(n_images=1)
2022-09-20 23:42:31.310037: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2022-09-20 23:42:32.874717: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8005

It's like outputs its inputs, let's train it and see if we can improve the model

def fit(epochs=100, chunk=10):
    for epoch in tqdm(range(1,epochs+1),desc="Training"):
        for X_b in train_ds:
            
            # Generate Labels
            real_labels, fake_labels = generate_labels()
            
            # Generate Fake Images
            noise = tf.random.normal(shape=X_b.shape)
            gen_out = generator.predict(noise)
            
            # Train Discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X_b, real_labels)
            discriminator.train_on_batch(gen_out, fake_labels)
            
            # Train Generator
            discriminator.trainable = False
            gan.train_on_batch(noise, [X_b, real_labels])
        if epoch%chunk==0:
            generator.save("Caleb-Face-Generator.h5")
            show_predictions(n_images=2)
            
Training
fit(epochs=10, chunk=2)
fit(epochs=10, chunk=2)
fit(epochs=50, chunk=10)
fit(epochs=100, chunk=20)
Evaluation
plt.figure(figsize=(20,10))
for i in range(10):
    noise = np.random.uniform(size=(1,256,256,3))
    gen_face = generator.predict(noise)[0]
    plt.subplot(2,5,i+1)
    show_image(gen_face)
plt.show()


