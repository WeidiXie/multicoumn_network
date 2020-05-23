from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Input, Conv1D
from face_resnet50 import ResNet50_base


def multiview_network_predict_step1(input_dim=(224, 224, 3)):

    # Deal with inputs.
    inputs = Input(shape=input_dim, name='img_a')
    #
    share_model = ResNet50_base(input_dim=input_dim)
    [feature, quality] = share_model(inputs)
    model = Model(inputs=inputs, outputs=[feature, quality])
    return model


def multiview_network_predict_step2(input_dim=(1, 4096)):
    # Deal with inputs.
    inputs = Input(shape=input_dim, name='concatenate_dim')
    #
    relation_atts = Conv1D(filters=1, kernel_size=(1,), activation='sigmoid', name='within_attention')(inputs)
    model = Model(inputs=inputs, outputs=relation_atts)
    return model