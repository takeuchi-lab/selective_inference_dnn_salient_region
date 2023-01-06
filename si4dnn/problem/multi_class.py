import tensorflow as tf

def construct_eta(output,region1,region2):
    output_argmax = tf.argmax(output,axis=3)

    index_region1 = output_argmax==region1
    index_region2 = output_argmax==region2
    tf.print(index_region1)
    tf.print(index_region2)

    n_region1 = tf.cast(tf.math.count_nonzero(index_region1),dtype=tf.float64)
    n_region2 = tf.cast(tf.math.count_nonzero(index_region2),dtype=tf.float64)

    assert n_region1.numpy() > 0
    assert n_region2.numpy() > 0

    e_region1 = tf.where(index_region1,tf.constant(1.0,dtype=tf.float64),0.0)
    e_region2 = tf.where(index_region2,tf.constant(1.0,dtype=tf.float64),0.0)

    eta = e_region1/n_region1 - e_region2/n_region2
    eta = tf.reshape(eta,-1)
    tf.print(eta.shape)

    return eta

def comparison_model(output_A,output_B):
    model_A = tf.argmax(output_A,axis=3)
    model_B = tf.argmax(output_B,axis=3)
    return tf.reduce_all(tf.math.equal(model_A, model_B))
