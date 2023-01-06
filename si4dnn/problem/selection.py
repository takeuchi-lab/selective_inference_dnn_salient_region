import tensorflow as tf
import numpy as np

class NoObjectsError(Exception):
    pass

# TODO モデルが作れなかったら例外なげろ!!
def construct_eta_cam_thr(model,X,X_ref,thr=0):
    # 注目したところだけ0,ほかは1
    Y_vec = tf.reshape(model, [-1])
    Y_plus = tf.cast(tf.where(Y_vec >= thr, 1, 0), dtype=tf.float64)
    n_plus = tf.reduce_sum(Y_plus)

    if n_plus == 0:
        raise NoObjectsError("オブジェクトがありません")
    else:
        eta = Y_plus / n_plus

    eta_new  = tf.concat([eta,-eta],axis=0)

    return eta_new

def construct_model(output,X):
    # 注目したところだけ0,ほかは1
    Y_vec = tf.reshape(output, [-1])
    Y_plus = tf.cast(tf.where(Y_vec >= 0, 1, 0), dtype=tf.float64)
    n_plus = tf.reduce_sum(Y_plus)


    if n_plus == 0:
        raise NoObjectsError("オブジェクトがありません")
    
    return Y_plus

def construct_eta_cam_thr_abs(output,X,X_ref,thr=0):

    X_diff = X-X_ref

    model = tf.reshape(output,-1)>=thr
    X_abs = tf.where(tf.reshape(X_diff,-1)>= 0,1,-1)
    eta = tf.where(model,X_abs,0)

    eta_new  = tf.concat([eta,-eta],axis=0)

    if tf.reduce_sum(tf.abs(eta_new)) == 0:
        raise NoObjectsError("オブジェクトがありません")

    return tf.cast(eta_new,dtype=tf.float64)

def comparison_model_cam_thr_abs(output_A,output_B,X_A,X_B,X_ref_A,X_ref_B,thr=0):

    X_diff_A = X_A - X_ref_A
    X_diff_B = X_B - X_ref_B

    model_A = tf.reshape(output_A,-1) >= thr
    model_B = tf.reshape(output_B,-1) >= thr

    X_abs_A = tf.where(tf.reshape(X_diff_A,-1)>=0,1,-1)
    X_abs_B = tf.where(tf.reshape(X_diff_B,-1)>=0,1,-1)

    model_abs_A = tf.where(model_A,X_abs_A,0)
    model_abs_B = tf.where(model_B,X_abs_B,0)

    return tf.reduce_all(tf.math.equal(model_abs_A,model_abs_B))

def construct_eta_each_element(model,elements_index,X):
    # 注目したところだけ0,ほかは1

    eta = np.zeros(model.shape[0])
    eta[elements_index] = 1

    return tf.constant(eta,dtype=tf.float64)

def comparison_model_thr(output_A,output_B,X_A,X_B,X_ref_A,X_ref_B,thr=0):
    model_A = output_A >= thr
    model_B = output_B >= thr
    return tf.reduce_all(tf.math.equal(model_A, model_B))

def construct_eta_top_k(model,k=10):
    # 注目したところだけ0,ほかは1
    Y_vec = tf.reshape(model, [-1])
    thr = tf.math.top_k(Y_vec,k=10)[-1]
    Y_plus = tf.cast(tf.where(Y_vec >= thr, 1, 0), dtype=tf.float64)
    eta = Y_plus / tf.reduce_sum

    return eta

def comparison_model_top_k(output_A,output_B,X_A,X_B,k=10):
    selected_indices_A = tf.math.top_k(tf.reshape(output_A,[-1]),k=10)
    selected_indices_B = tf.math.top_k(tf.reshape(output_B,[-1]),k=10)
    return tf.reduce_all(tf.math.equal(selected_indices_A, selected_indices_B))
