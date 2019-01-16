import tensorflow as tf
import numpy as np

def gen_model():
    'TODO: docstring'    
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=20,
                               kernel_size=5),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=50,
                               kernel_size=5),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])

def gen_mnist_iterator(x, y, bs):
    'TODO: docstring'    
    x = (x / 255.0).astype(np.float32)[..., tf.newaxis]
    y = tf.one_hot(y, 10)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(x.shape[0]).batch(bs).repeat()
    return ds.make_one_shot_iterator()
