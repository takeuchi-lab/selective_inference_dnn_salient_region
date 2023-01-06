import numpy as np
import tensorflow as tf
from scipy import stats
from .si import si
from .problem import bin_class
from .problem import multi_class

def si4dnn_binary_segmentation(model):
    return si.SI4DNN(model,bin_class.construct_eta,bin_class.comparison_model)

def si4dnn_multi_class_segmentation(model,region1,region2):
    return si.SI4DNN(model,lambda x:multi_class.construct_eta(x,region1,region2),multi_class.comparison_model)