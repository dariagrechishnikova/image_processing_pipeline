# -*- coding: utf-8 -*-
"""main_imseg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sx9K9qgmN7WmVUVQMINmcAR7PPVIDiR8
"""

#Setup the enviroment on colab
#Google drive should be mounted to access folder with deduplicated images ('dedup')
#Two folders model_checkpoints/ and network_info/ must be located on Google drive
!unzip /content/drive/MyDrive/FetReg.zip
!unzip /content/FetReg2021_Task1_Segmentation.zip

!mkdir label
!mkdir /content/tensor_board_logs/


sh = """
for filename in /content/Video*; do
  echo $filename
  cp $filename/labels/* /content/label
done
"""
with open('script.sh', 'w') as file:
  file.write(sh)

!bash script.sh

!pip install -q -U albumentations
!echo "$(pip freeze | grep albumentations) is successfully installed"

!pip install iterative-stratification
!pip install -U segmentation-models

# Commented out IPython magic to ensure Python compatibility.
import sys
sys.path.append('/content/drive/MyDrive/code_pipeline')


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
from dataclasses import dataclass
from runner import *
from trainer import *
from initial_parser import *
from splitter import *
from data_provider import *
from custom_models import *
from custom_metrics import *
from custom_losses import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
import random
from tensorflow.keras.layers.experimental import preprocessing 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from tensorflow.keras.applications import EfficientNetB0
from keras import backend as K
from keras.layers.merge import concatenate
from keras.utils.data_utils import get_file
from tensorflow import keras
import segmentation_models as sm
import cv2
import datetime
from segmentation_models.losses import bce_jaccard_loss
import pickle
from sklearn.model_selection import train_test_split


# %load_ext tensorboard
log_dir = "tensor_board_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# %tensorboard --logdir tensor_board_logs/

def main_imseg(
    initial_split,
    data_path,
    label_path,
    class_count,
    test_size,
    val_size,
    seed,
    cv_folds,
    img_size,
    buffer_size,
    batch_size,
    model_obj,
    optimizer,
    loss,
    metrics,
    epochs,
    log_name,
    tr_val_split_file, 
    test_file,
    net_info_path, 
    model_checkpoints_path,
    callbacks = []
    ):
  initial_parser_obj = initial_parser_imseg(data_path, label_path, class_count)
  splitter_obj = splitter_imseg(test_size, val_size, seed, cv_folds, tr_val_split_file, test_file)
  model = model_obj.build_model()
  data_provider_obj = data_provider_imseg(data_path, label_path, img_size, 
                                          class_count, buffer_size, batch_size)
  trainer_obj = trainer_imseg(model, optimizer, loss, metrics,
               batch_size, epochs, callbacks, log_name, net_info_path, model_checkpoints_path)
  
  if initial_split == 'Y':
    runner_with_split(initial_parser_obj, splitter_obj, data_provider_obj, trainer_obj)
  else:
    with open(tr_val_split_file, 'rb') as f:
      train_val_splits_from_file = pickle.load(f)
      for tr, val in train_val_splits_from_file:
        print(len(tr), len(val))
        print(tr[:10])
    runner(train_val_splits_from_file, initial_parser_obj, splitter_obj, data_provider_obj, trainer_obj)

params_dict = {'test_model' : {'initial_split': 'N','batch_size': 32, 'buffer_size': 1000, 'callbacks': [],  'class_count': 4, 'cv_folds': 5, 'data_path': '/content/drive/MyDrive/dedups/', 'epochs': 32, 'img_size': 224, 'label_path': '/content/label/', 'loss': iou_fn_loss, 'metrics': [jacard_coef, mean_iou], 'model_obj': efficientnet_bb_unet(output_mask_channels = 4), 'optimizer': 'adam', 'seed': 0, 'test_size': 0.2, 'val_size': 0.1, 'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split', 'test_file': '/content/drive/MyDrive/models_misc/test_split', 'net_info_path': '/content/drive/MyDrive/network_info/', 'model_checkpoints_path': '/content/drive/MyDrive/model_checkpoints'}}

for model_id in params_dict:
  print(model_id)
  main_imseg(log_name = model_id, **params_dict[model_id])





# Run for each class label separetly. Class label is given to splitter_obj as augment inside main function.
def main_imseg_binary_from_multiclass_cl1(
    initial_split,
    data_path,
    label_path,
    class_count,
    test_size,
    val_size,
    seed,
    cv_folds,
    img_size,
    buffer_size,
    batch_size,
    model_obj,
    optimizer,
    loss,
    metrics,
    epochs,
    log_name,
    tr_val_split_file, 
    test_file,
    net_info_path, 
    model_checkpoints_path,
    class_num,
    callbacks = []
    ):
  initial_parser_obj = initial_parser_imseg(data_path, label_path, class_count)
  splitter_parent_obj = splitter_imseg(test_size, val_size, seed, cv_folds, tr_val_split_file, test_file)
  splitter_obj = splitter_imseg_binary_from_multiclass(splitter_parent_obj, initial_parser_obj, class_num)
  model = model_obj.build_model()
  data_provider_obj = data_provider_imseg(data_path, label_path, img_size, 
                                          class_count, buffer_size, batch_size)
  trainer_obj = trainer_imseg(model, optimizer, loss, metrics,
               batch_size, epochs, callbacks, log_name, net_info_path, model_checkpoints_path)
  
  if initial_split == 'Y':
    runner_with_split(initial_parser_obj, splitter_obj, data_provider_obj, trainer_obj)
  else:
    with open(tr_val_split_file, 'rb') as f:
      train_val_splits_from_file = pickle.load(f)
      for tr, val in train_val_splits_from_file:
        print(len(tr), len(val))
        print(tr[:10])
    runner(train_val_splits_from_file, initial_parser_obj, splitter_obj, data_provider_obj, trainer_obj)

params_dict = {'test_model' : {'initial_split': 'Y', 'class_num' : 1, 'batch_size': 32, 'buffer_size': 1000, 'callbacks': [],  'class_count': 4, 'cv_folds': 5, 'data_path': '/content/drive/MyDrive/dedups/', 'epochs': 32, 'img_size': 224, 'label_path': '/content/label/', 'loss': iou_fn_loss, 'metrics': [jacard_coef, mean_iou], 'model_obj': efficientnet_bb_unet(output_mask_channels = 4), 'optimizer': 'adam', 'seed': 0, 'test_size': 0.2, 'val_size': 0.1, 'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints'}}

for model_id in params_dict:
  print(model_id)
  main_imseg_binary_from_multiclass_cl1(log_name = model_id, **params_dict[model_id])







from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()


paths = Path('/content/drive/MyDrive/network_info').glob('**/*.csv')



a = 3  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(14,10))

for path in paths:
  data = pd.read_csv(path, sep=',')
  axes = plt.subplot(a, b, c)
  plt.plot(data['epoch'], scaler.fit_transform(data[['loss']]),  label="loss")
  plt.plot(data['epoch'], scaler.fit_transform(data[['val_loss']]),  label="val_loss")
  plt.title(path.stem)
  #plt.xlabel('epoch')
  handles, labels = axes.get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center')
  c = c + 1
  plt.grid()

plt.show()