import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tqdm
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
import argparse
import sys
from keras.layers import Layer
from keras import backend as K
from keras import activations, initializers, regularizers
import numpy as np
import imageio.v2 as sci
import glob
from sklearn.model_selection import KFold
import time
import random
import numpy as np
import cv2
from keras_visualizer import visualizer 

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=0.0005, type=float)
    parser.add_argument('--useGated', dest='useGated',
                        help='use Gated Attention',
                        default=True, type=int)
    args = parser.parse_args()
    return args

args = parse_args()

#CREATE CUSTOM MIL LAYER
class Mil_Attention(Layer):
    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                    use_bias=True, use_gated=False, **kwargs):
        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.v_init = initializers.get(kernel_initializer)
        self.w_init = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)


        self.v_regularizer = regularizers.get(kernel_regularizer)
        self.w_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Mil_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.V = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)


        self.w = self.add_weight(shape=(self.L_dim, 1),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)


        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True


    def call(self, x, mask=None):
        n, d = x.shape
        ori_x = x
        # do Vhk^T
        x = K.tanh(K.dot(x, self.V)) # (2,64)

        if self.use_gated:
            gate_x = K.sigmoid(K.dot(ori_x, self.U))
            ac_x = x * gate_x
        else:
            ac_x = x

        # do w^T x
        soft_x = K.dot(ac_x, self.w)  # (2,64) * (64, 1) = (2,1)
        alpha = K.softmax(K.transpose(soft_x)) # (2,1)
        alpha = K.transpose(alpha)
        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'v_initializer': initializers.serialize(self.V.initializer),
            'w_initializer': initializers.serialize(self.w.initializer),
            'v_regularizer': regularizers.serialize(self.v_regularizer),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Mil_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Custom_Pooling(Layer):
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Custom_Pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid #out = x for attention weighted vector for concat #K.sigmoid(x)
        out = x

        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Custom_Pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


input_dim = (512, 512, 3)

def OCCRiskNet (input_dim, args):
    tower_0 = layers.Conv2D(64, 1, padding="same", activation="relu")(input_dim)
    tower_1 = layers.Conv2D(64, 1, padding="same", activation="relu")(input_dim)
    tower_1 = layers.Conv2D(64, 3, padding="same", activation="relu")(tower_1)
    tower_2 = layers.Conv2D(64, 1, padding="same", activation="relu")(input_dim)
    tower_2 = layers.Conv2D(64, 5, padding="same", activation="relu")(tower_2)
    tower_3 = layers.MaxPool2D((3,3),strides=(1,1), padding="same")(input_dim)
    tower_3 = layers.Conv2D(64, 1, padding="same", activation="relu")(tower_3)      
    inception = tf.keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
    model = layers.Conv2D(128, 3, activation = "relu")(inception)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.Conv2D(128, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.MaxPool2D()(model)
    model = layers.Conv2D(128, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.Conv2D(128, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.MaxPool2D()(model)
    model = layers.Conv2D(256, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.Conv2D(256, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.MaxPool2D()(model)
    model = layers.Conv2D(512, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.MaxPool2D()(model)
    model = layers.Conv2D(512, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.MaxPool2D()(model)
    model = layers.Conv2D(512, 3, activation = "relu")(model)
    model= tf.keras.layers.BatchNormalization()(model)
    model = layers.MaxPool2D()(model)

    return model

#inputs

input1 = keras.Input(shape=(512, 512, 3))
input2 = keras.Input(shape=(512, 512, 3))
input3 = keras.Input(shape=(512, 512, 3))

#models

model1 = OCCRiskNet(input1, args)
model2 = OCCRiskNet(input2, args)
model3 = OCCRiskNet(input3, args)

#obtaining low feature dimensional feature map
ld_f_map_HE = layers.GlobalAveragePooling2D()(model1)
ld_f_map_p53 = layers.GlobalAveragePooling2D()(model2)
ld_f_map_PDPN = layers.GlobalAveragePooling2D()(model3)

#ATTN MIL Layer
alpha_HE = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=keras.regularizers.l2(0.0005), name='self-attn-weight_HE', use_gated=args.useGated)(ld_f_map_HE)
alpha_p53 = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=keras.regularizers.l2(0.0005), name='self-attn-weight_p53', use_gated=args.useGated)(ld_f_map_p53)
alpha_PDPN = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=keras.regularizers.l2(0.0005), name='self-attn-weight_PDPN', use_gated=args.useGated)(ld_f_map_PDPN)

#multiply weights with feature map
mul_HE = layers.multiply([alpha_HE, ld_f_map_HE])
mul_p53 = layers.multiply([alpha_p53, ld_f_map_p53])
mul_PDPN = layers.multiply([alpha_PDPN, ld_f_map_PDPN])

#Weighted feature vector in each channel
wv1 = Custom_Pooling(output_dim=1, name='Weighted_vector_bag_HE')(mul_HE)
wv2 = Custom_Pooling(output_dim=1, name='Weighted_vector_bag_p53')(mul_p53)
wv3 = Custom_Pooling(output_dim=1, name='Weighted_vector_bag_PDPN')(mul_PDPN)

#concatenate weighted feature vector
glob_feat_vector = tf.keras.layers.concatenate([wv1, wv2, wv3], axis = -1)

#Output prediction
output = layers.Dense(1, activation='sigmoid')(glob_feat_vector)

#summary

model = tf.keras.Model(inputs = [input1, input2, input3], outputs = output)
opt = keras.optimizers.Adam()
model.compile(optimizer=opt,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
model._name = "OCCRiskNet"
model.summary()

with open('modelsummary.txt', 'w') as f:

    model.summary(print_fn=lambda x: f.write(x + '\n'))



    
    
