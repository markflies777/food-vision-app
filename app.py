import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import streamlit as st


def load_model(filename):
    model = tf.keras.models.load_model(filename)
    return model


def preprocessing_img(upload_image):
    pic = mpimg.imread(upload_image)
    resized_image = tf.image.resize(pic, [224, 224])
    return resized_image, pic


def make_prediction(upload_image, model):
    resized_image, pic = preprocessing_img(upload_image)
    prediction = model.predict(tf.expand_dims(resized_image, axis=0))[0]
    # Find the index of the top 5
    top_5 = prediction.argsort()[::-1][:5]
    # Find the probability of the top 5
    top_5_prob = np.sort(prediction)[::-1][:5]
    return pic, top_5, top_5_prob


