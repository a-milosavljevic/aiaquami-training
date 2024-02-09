"""
model.py script is by traing.py to create the model.
You do not have to run this script.
"""
from data import *


if architecture == 'ResNet50':
    preprocess_input = tf.keras.applications.resnet.preprocess_input
    encoder = tf.keras.applications.ResNet50
    encoder_last_conv_layer = 'conv5_block3_out'
elif architecture == 'ResNet50V2':
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    encoder = tf.keras.applications.ResNet50V2
    encoder_last_conv_layer = 'post_relu'
elif architecture == 'EfficientNetB0':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetB0
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetB1':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetB1
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetB2':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetB2
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetB3':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetB3
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetB4':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetB4
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetB5':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetB5
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetB6':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetB6
    encoder_last_conv_layer = 'top_activation'

'''
elif architecture == 'EfficientNetV2B0':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetV2B0
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetV2B1':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetV2B1
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetV2B2':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetV2B2
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetV2B3':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetV2B3
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetV2S':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetV2S
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetV2M':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetV2M
    encoder_last_conv_layer = 'top_activation'
elif architecture == 'EfficientNetV2L':
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    encoder = tf.keras.applications.EfficientNetV2L
    encoder_last_conv_layer = 'top_activation'
'''

def create_model(trainable_encoder):
    ishp = (image_size, image_size, 3)
    x = tf.keras.layers.Input(shape=ishp, name='input')
    y = tf.keras.layers.Lambda(preprocess_input)(x)

    backbone = encoder(include_top=False, weights='imagenet',
                       input_shape=ishp, input_tensor=y, pooling='avg', classes=len(classes))

    if not trainable_encoder:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    y = backbone.output

    if dropout_rate > 0:
        y = tf.keras.layers.Dropout(rate=dropout_rate)(y)

    if hidden_neurons > 0:
        y = tf.keras.layers.Dense(hidden_neurons, activation='relu', name='hidden')(y)
        if dropout_rate > 0:
            y = tf.keras.layers.Dropout(rate=dropout_rate)(y)

    y = tf.keras.layers.Dense(len(classes), activation='softmax', name='output')(y)

    model = tf.keras.models.Model(inputs=x, outputs=y)

    return model
