import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.models import Model

def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder path
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)

    # Decoder path
    up5 = UpSampling3D(size=(2, 2, 2))(conv4)
    up5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=-1)
    conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge5)

    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=-1)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=-1)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge7)

    outputs = Conv3D(3, (1, 1, 1), activation='linear')(conv7)

    model = Model(inputs=inputs, outputs=outputs)

    return model
