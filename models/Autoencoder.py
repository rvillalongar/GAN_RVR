# Libreria la cual genera autoencoders,
# segun la configuracion de la red conv que uno esta entregando
# Un autoencoder corresponde a 2 modelos conectados que son espejos 
# en su configuracion

from tensorflow.keras.layers import Input, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, Conv2D
from tensorflow.keras.models import Model
from keras import backend as K

import numpy as np
import json 
import os 
import pickle


class Autoencoders():
    """

    """
    def __init__(self
    , input_dim
    , encoder_conv_filters
    , encoder_conv_kernel_size
    , encoder_conv_strides
    , decoder_conv_t_filters
    , decoder_conv_t_kernel_size
    , decoder_conv_t_strides
    , z_dim
    , use_batch_norm = False
    , use_dropout = False
    ):

        self.name = 'autoecoder'
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size= encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size=decoder_conv_t_kernel_size
        self.decoder_conv_t_strides=decoder_conv_t_strides
        self.z_dim=z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        self.n_layers_encoder= len(encoder_conv_filters)
        self.n_layers_decoder=len(decoder_conv_t_filters)

        self.build()
    
    def build(self):
        # Creación del Encoder

        encoder_input = Input(shape= self.input_dim, name='Entrada_Encoder')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters= self.encoder_conv_filters[i],
                kernel_size= self.encoder_conv_kernel_size[i],
                strides= self.encoder_conv_strides[i],
                padding='same',
                name = 'encoder_conv_' + str(i)
            )
            x = conv_layer(x)
            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)
            if self.use_dropout:
                x = Dropout()(x)
        
        shape_before_flattening = K.int_shape(x)[1:]
        #No se por que chucha es de 1..n el flatering

        print(shape_before_flattening)

        x = Flatten()(x)
        encoder_output = Dense(self.z_dim, name ='encoder_output')(x)

        self.encoder = Model(encoder_input, encoder_output)

        #Creacion del Decoder
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i]
                , kernel_size=self.decoder_conv_t_kernel_size[i]
                , strides=self.decoder_conv_t_strides[i]
                , padding='same'
                , name='decoder_conv_t_' + str(i)
            )
        
        x = conv_t_layer(x)

        if i<self.n_layers_decoder -1:
            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)
            
            if self.use_dropout:
                x = Activation('sigmoid')(x)
        
        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ## Construccion de full Autoencoder
        model_input = encoder_input
        model_output = self.decoder(encoder_input)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        

            




