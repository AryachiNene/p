import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

class_number = 2  # 分类数

#编码部分
#使用VGG网络提取输入图像的特征
def encoder(input_height, input_width):

    img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f1 = x

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f2 = x

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f3 = x

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f4 = x

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def decoder(feature_map_list, class_number, input_height=512, input_width=512, encoder_level=3):
    #从Encoder中获取特征图
    feature_map = feature_map_list[encoder_level]

    x = ZeroPadding2D((1, 1))(feature_map)
    x = Conv2D(512, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Conv2D(class_number, (3, 3), padding='same')(x)
    x = Reshape((int(input_height / 2) * int(input_width / 2), -1))(x)
    output = Softmax()(x)

    return output


def main(Height=512, Width=512):

    img_input, feature_map_list = encoder(input_height=Height, input_width=Width)

    output = decoder(feature_map_list, class_number=class_number, input_height=Height, input_width=Width,
                     encoder_level=3)

    model = Model(img_input, output)

    return model

if __name__ == '__main__':
    main(Height=512, Width=512)
