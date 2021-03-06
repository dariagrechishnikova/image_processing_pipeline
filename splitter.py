# -*- coding: utf-8 -*-
"""splitter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lJ5SX8ALF2zf0xFUg-_ojYrpQjcGwSE4
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
from dataclasses import dataclass
from runner import *
from trainer import *
from initial_parser import *
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

class splitter_imseg():
  def __init__(self, input_test_size, input_val_size, input_seed, input_cv_folds, input_tr_val_split_file, input_test_file):
      self.test_size = input_test_size
      self.val_size = input_val_size
      self.seed = input_seed
      self.cv_folds = input_cv_folds
      self.tr_val_split_file = input_tr_val_split_file
      self.test_file = input_test_file


  def train_test_split(self, item_list):
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=self.test_size, random_state=self.seed)
    data, labels = item_list[0], item_list[1]
    for indx_train, indx_test in msss.split(data, labels):
      train_items = [data[i] for i in indx_train]
      test_items = [data[i] for i in indx_test]
      train_labels = [labels[i] for i in indx_train]
      test_labels = [labels[i] for i in indx_test]
      break
    return train_items, train_labels, test_items, test_labels


  def train_val_split(self, ds_items, ds_labels):
    mskf = MultilabelStratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
    folds = []
    for indx_train, indx_val in mskf.split(ds_items, ds_labels):
      train_part = [ds_items[i] for i in indx_train]
      val_part = [ds_items[i] for i in indx_val]
      folds.append((train_part, val_part))
    return folds

  def get_split(self, item_list):
    train_ds, train_lbls, test_ds, test_lbls = self.train_test_split(item_list)
    folds_out = self.train_val_split(train_ds, train_lbls)
    with open(self.tr_val_split_file, 'wb') as f:
      pickle.dump(folds_out, f)
    with open(self.test_file, 'wb') as f:
      pickle.dump(test_ds, f)
    return folds_out, test_ds






class splitter_imseg_binary_from_multiclass():
  def __init__(self, input_splitter_imseg_obj, input_initial_parser_obj, input_class_num):
      self.parent_splitter = input_splitter_imseg_obj
      self.init_parser = input_initial_parser_obj
      self.class_num = input_class_num


  def filenames_per_class(self, train_part_names, train_part_labels):
    #cl = self.init_parser.class_labels(train_part)
    out = []
    for cl_item, file_nm in zip(train_part_labels, train_part_names):
      if cl_item[self.class_num] == 1:
        out.append(file_nm)
    return out


  def get_split(self, item_list):
    train_ds, train_lbls, test_ds, test_lbls = self.parent_splitter.train_test_split(item_list)
    ds = self.filenames_per_class(train_ds, train_lbls)
    folds_out = [train_test_split(ds, test_size=0.2, random_state=self.parent_splitter.seed)]
    with open(self.parent_splitter.tr_val_split_file, 'wb') as f:
      pickle.dump(folds_out, f)
    with open(self.parent_splitter.test_file, 'wb') as f:
      pickle.dump(test_ds, f)
    return folds_out, test_ds