import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

model = tf.keras.models.load_model('best_model_postx.keras')

train_score = model.evaluate(tr_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(ts_gen, verbose=1)

print(f"Train Loss: {train_score[0]:.4f} | Train Accuracy: {train_score[1]*100:.2f}%")
print(f"Validation Loss: {valid_score[0]:.4f} | Validation Accuracy: {valid_score[1]*100:.2f}%")
print(f"Test Loss: {test_score[0]:.4f} | Test Accuracy: {test_score[1]*100:.2f}%")


preds = model.predict(ts_gen)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(ts_gen.classes, y_pred)
labels = list(tr_gen.class_indices.keys())

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

clr = classification_report(ts_gen.classes, y_pred)
print(clr)
