import tensorflow as tf

# TODO モデルが作れなかったら例外なげろ!!
def construct_eta(model):
    tf.print(model)
    Y_vec = tf.reshape(model, [-1])
    Y_plus = tf.cast(tf.where(Y_vec >= 0.5, 1, 0), dtype=tf.float64)
    n_plus = tf.reduce_sum(Y_plus)
    Y_minus = tf.cast(tf.where(Y_vec < 0.5, 1, 0), dtype=tf.float64)
    n_minus = tf.reduce_sum(Y_minus)

    if n_plus == 0:
        assert False
        eta = Y_minus / n_minus
    elif n_minus == 0:
        assert False
        eta = Y_plus / n_plus
    else:
        eta = (Y_plus / n_plus) - (Y_minus / n_minus)

    return eta

def comparison_model(output_A,output_B):
    model_A = output_A >= 0.5
    model_B = output_B >= 0.5
    return tf.reduce_all(tf.math.equal(model_A, model_B))
