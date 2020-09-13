import numpy as np
import random
from audiopre import *
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

np.random.seed(123)
random.seed(123)


def clipped_relu(inp):
        relu = Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inp)
        return relu

def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = f'res{stage}_{block}_branch'

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '_2b')(x)
    x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)

    x = clipped_relu(x)

    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x

def conv_and_res_block(inp, filters, stage):
    conv_name = 'conv{}-s'.format(filters)
    # TODO: why kernel_regularizer?
    o = Conv2D(filters,
               kernel_size=5,
               strides=2,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name)(inp)
    o = BatchNormalization(name=conv_name + '_bn')(o)
    o = clipped_relu(o)
    for i in range(3):
        o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
    return o

def cnn_component(inp):
    x = conv_and_res_block(inp, 64, stage=1)
    x = conv_and_res_block(x, 128, stage=2)
    x = conv_and_res_block(x, 256, stage=3)
    x = conv_and_res_block(x, 512, stage=4)
    return x


inp = Input(batch_shape= (None,160,64,1) , name='input')
x = cnn_component(inp)
x = Reshape((-1, 2048))(x)
x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
x = Dense(512, name='affine')(x)
x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
model = Model(inp, x, name='ResCNN')

model.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

np.random.seed(123)
random.seed(123)
mfcc_001 = sample_from_mfcc(read_mfcc('ad1.wav.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('PhilippeRemy_003.wav', SAMPLE_RATE), NUM_FRAMES)

predict_001 = model.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.predict(np.expand_dims(mfcc_002, axis=0))

mul = np.multiply(predict_001, predict_002)
s = np.sum(mul, axis=1)
s

 
