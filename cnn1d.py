import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import merge
from keras.layers import concatenate
from keras.layers import Reshape
from keras.models import Model

from ssd_layers import Normalize
from ssd_layers import PriorBox

def cnn1d(input_shape):
    """CNN1d architecture.
    """
    net = {}
    # Block 1
    input_tensor = Input(shape=input_shape)
    tracedata_size = (input_shape[0])
    net['input'] = input_tensor
    net['conv1_1'] = Conv1D(16, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Conv1D(16, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling1D(2, strides=None, name='pool1')(net['conv1_2'])
    # Block 2
    net['conv2_1'] = Conv1D(32, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Conv1D(32, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling1D(2, strides=None, name='pool2')(net['conv2_2'])
    # Block 3
    net['conv3_1'] = Conv1D(64, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Conv1D(64, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv1D(64, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling1D(2, strides=None, name='pool3')(net['conv3_3'])
    # Block 4
    net['conv4_1'] = Conv1D(128, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Conv1D(128, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv1D(128, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling1D(2, strides=None, name='pool4')(net['conv4_3'])
    # Block 5
    net['conv5_1'] = Conv1D(256, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Conv1D(256, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv1D(256, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling1D(2, strides=None, name='pool5')(net['conv5_3'])

    # Block 6
    net['conv6_1'] = Conv1D(512, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv6_1')(net['pool5'])
    net['conv6_2'] = Conv1D(512, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv6_2')(net['conv6_1'])
    net['conv6_3'] = Conv1D(512, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv6_3')(net['conv6_2'])
    net['pool6'] = MaxPooling1D(2, strides=None, name='pool6')(net['conv6_3'])


    # upsampling

    # Block 7
    net['upsmpl7'] = UpSampling1D(2)(net['pool6'])
    net['conv7_1'] = Conv1D(256, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv7_1')(net['upsmpl7'])

    # Block 8
    net['upsmpl8'] = UpSampling1D(2)(net['conv7_1'])
    net['conv8_1'] = Conv1D(128, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv8_1')(net['upsmpl8'])

    # Block 9
    net['upsmpl9'] = UpSampling1D(2)(net['conv8_1'])
    net['conv9_1'] = Conv1D(64, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv9_1')(net['upsmpl9'])

    # Block 10
    net['upsmpl10'] = UpSampling1D(2)(net['conv9_1'])
    net['conv10_1'] = Conv1D(32, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv10_1')(net['upsmpl10'])

    # Block 11
    net['upsmpl11'] = UpSampling1D(2)(net['conv10_1'])
    net['conv11_1'] = Conv1D(16, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv11_1')(net['upsmpl11'])
    # predictions
    net['predictions'] = Conv1D(1, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='predictions')(net['conv11_1'])                                                                      
    
    model = Model(net['input'], net['predictions'])
    return model