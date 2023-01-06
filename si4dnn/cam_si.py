import numpy as np
import tensorflow as tf
from scipy import stats
from .si import si
from .problem import selection

def si4dnn_cam_si_thr(model,thr=0):
    construct_eta = lambda a,b,c:selection.construct_eta_cam_thr(a,b,c,thr=thr)
    comparison_model = lambda a,b,c,d,e,f:selection.comparison_model_thr(a,b,c,d,e,f,thr=thr)
    return si.SI4DNN(model,construct_eta,construct_eta,comparison_model)

def si4dnn_cam_si_thr_abs(model,thr=0):
    construct_eta = lambda a,b,c:selection.construct_eta_cam_thr_abs(a,b,c,thr=thr)
    comparison_model = lambda a,b,c,d,e,f:selection.comparison_model_cam_thr_abs(a,b,c,d,e,f,thr=thr)
    return si.SI4DNN(model,construct_eta,construct_eta,comparison_model,abs=True)

def si4dnn_cam_si_top_k(model):
    return si.SI4DNN(model,selection.construct_eta_each_element,selection.construct_eta_thr,selection.comparison_model_thr,k=10)