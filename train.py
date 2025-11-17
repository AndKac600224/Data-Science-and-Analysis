import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import warnings

from tensorflow.python.keras.saving.save import load_model

warnings.filterwarnings("ignore")


def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])

    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df

def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])

    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

def valid_df(vld_path):
    classes, class_paths = zip(*[(label, os.path.join(vld_path, label, image))
                                 for label in os.listdir(vld_path) if os.path.isdir(os.path.join(vld_path, label))
                                 for image in os.listdir(os.path.join(vld_path, label))])

    vld_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return vld_df

tr_df = train_df('dataset/training')
ts_df = test_df('dataset/testing')
vld_df = valid_df('dataset/validation')

''' ------------------------------------------------------------- '''

_gen = ImageDataGenerator(rescale=1/255,
                          brightness_range=(0.8, 1.2))

ts_gen = ImageDataGenerator(rescale=1/255)


tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path',
                                  y_col='Class', batch_size=32,
                                  target_size=(224,224),
                                  class_mode='categorical')

valid_gen = _gen.flow_from_dataframe(vld_df, x_col='Class Path',
                                     y_col='Class', batch_size=32,
                                     target_size=(224,224),
                                     class_mode='categorical')

ts_gen = ts_gen.flow_from_dataframe(ts_df, x_col='Class Path',
                                  y_col='Class', batch_size=16,
                                  target_size=(224,224), shuffle=False,
                                    class_mode='categorical')


class_dict = tr_gen.class_indices
classes = list(class_dict.keys())
''' ---------------------------------------------------------- '''
img_shape=(224,224,3)

model = Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(Adamax(learning_rate= 0.001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy',
                        Precision(),
                        Recall()])

callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model_postx.keras", save_best_only=True)

''' -------------------------------------------------------- '''

model.fit(
    tr_gen,
    validation_data=valid_gen,
    epochs=13,
    callbacks=[callback, checkpoint],
    verbose=1
)

## wytrenowany i zapisany do best_model_postx.keras

