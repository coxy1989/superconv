import math, multiprocessing, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from functools import partial

def gen_model():
    "TODO: docstring"
    return tf.keras.Sequential([InputLayer(input_shape=(32,32,3)),
                                Conv2D(filters=32,
                                kernel_size=5, 
                                padding='same',
                                kernel_initializer=tf.keras.initializers.RandomNormal(0, 1e-4)),
                                MaxPool2D(pool_size=3, strides=2),
                                ReLU(),
                                BatchNormalization(),
                                Conv2D(filters=32,
                                       kernel_size=5, 
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(0, 1e-2)),
                                ReLU(),
                                AveragePooling2D(pool_size=3, strides=2),
                                BatchNormalization(),
                                Conv2D(filters=64,
                                       kernel_size=5, 
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(0, 1e-2)),
                                ReLU(),
                                AveragePooling2D(pool_size=3, strides=2),
                                Flatten(),
                                Dense(10, activation='softmax')])

def parse_tfrecord(example):
    "TODO: docstring"
    example_fmt = {
        "image": tf.FixedLenFeature((), tf.string, ""),
        "label": tf.FixedLenFeature((), tf.int64, -1)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image.set_shape([3 * 32 * 32])
    image = tf.cast(
        tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(parsed['label'], tf.int32)
    return image, tf.one_hot(label, 10)

def gen_mean_img(tf_record_file):
    "TODO: docstring"
    ds = tf.data.TFRecordDataset(tf_record_file).map(parse_tfrecord).batch(1)
    it = ds.make_one_shot_iterator()
    next_elem = it.get_next()
    n = 0
    tot_img = np.zeros([32, 32, 3])
    with tf.Session() as sess:
        while True:
            try:
                image, _ = sess.run(next_elem)
                tot_img += image[0,...]
                n += 1
            except tf.errors.OutOfRangeError:
                return (tot_img/n), n

def count_elements(tf_record_file):
    "TODO: docstring"
    ds = tf.data.TFRecordDataset(tf_record_file).map(parse_tfrecord).batch(1)
    it = ds.make_one_shot_iterator()
    next_elem = it.get_next()
    n = 0
    with tf.Session() as sess:
        while True:
            try:
                image, _ = sess.run(next_elem)
                n += 1
            except tf.errors.OutOfRangeError:
                return n

def prepare_data(mean_image, tf_example):
    "TODO: docstring"
    image, label = parse_tfrecord(tf_example)
    image = (image - mean_image) / 255.0
    return image, label

def gen_iterator(filename, mean_image, bs, num_cpus):
    "TODO: docstring"
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.shuffle(100).repeat()
    ds = ds.map(num_parallel_calls=num_cpus, map_func=partial(prepare_data, mean_image))
    ds = ds.batch(bs).prefetch(1)
    return ds.make_one_shot_iterator()
