# -*- coding: utf-8 -*-
"""data_provider.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15zegQhVM2kYWQ1G61M-O3Zj_qm9x5EfJ
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
from dataclasses import dataclass
from runner import *
from trainer import *
from initial_parser import *
from splitter import *
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


class parser_imseg():
  def __init__(self, input_path_image, input_path_mask, input_img_size, input_classes_cnt):
      self.path_image = input_path_image
      self.path_mask = input_path_mask
      self.img_size = input_img_size
      self.classes_cnt = input_classes_cnt

  def convert_mask_to_confidence_tensor(self, mask):
    output_shape = (*mask.shape, self.classes_cnt)
    output = np.zeros(shape=output_shape)
    for class_label in range(self.classes_cnt):
      output[mask == class_label, class_label] = 1
    return np.ndarray.astype(output, np.float32)

  def create_masks(self, mask):
    mask = tf.squeeze(mask)
    msk_shape = [*mask.shape, self.classes_cnt]
    mask = tf.numpy_function(self.convert_mask_to_confidence_tensor, [mask], np.float32)
    mask.set_shape(msk_shape)
    return mask

  def parse_image(self, filename):
    image = tf.io.read_file(self.path_image + filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, [self.img_size, self.img_size])
    return image

  def parse_mask(self, filename):
    mask = tf.io.read_file(self.path_mask + filename)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [self.img_size, self.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if self.classes_cnt > 2:
      mask = self.create_masks(mask)
    return mask







class data_provider_imseg():
  def __init__(self, input_data_path, input_label_path, input_img_size, input_class_count, input_buffer_size, input_batch_size, input_parser):
      self.path_image = input_data_path
      self.path_mask = input_label_path
      self.img_size = input_img_size
      self.classes_cnt = input_class_count
      self.buffer_size = input_buffer_size
      self.batch_size = input_batch_size
      self.parser = input_parser


  def configure_train_ds(self, train_images_ds):
    augmentor = augmentor_imseg()
    train_batches = (
        train_images_ds
        .cache()
        .shuffle(self.buffer_size)
        .map(augmentor.tf_augment)
        .batch(self.batch_size)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    return train_batches

  def configure_val_ds(self, test_images_ds):
    return test_images_ds.batch(self.batch_size)


  def create_tf_dataset(self, index_list):
    random.shuffle(index_list)
    filenames_ds = tf.data.Dataset.from_tensor_slices(index_list)
    images_ds = filenames_ds.map(self.parser.parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    masks_ds = filenames_ds.map(self.parser.parse_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = tf.data.Dataset.zip((images_ds, masks_ds))
    return ds

  def get_train_tfset(self, index_list):
    return self.configure_train_ds(self.create_tf_dataset(index_list))

  def get_val_tfset(self, index_list):
    return self.configure_val_ds(self.create_tf_dataset(index_list))

class augmentor_imseg():
  def __init__(self):
    self.transforms = A.Compose([
                                 A.HorizontalFlip(p=0.5),
                                 A.VerticalFlip(p=0.5),
                                 A.RandomRotate90(p=0.5),
                                 A.Transpose(p=0.5),
                                 
                                 ])

  def aug_fn(self, image, mask):
    masks = [mask[:,:,0], mask[:,:,1], mask[:,:,2], mask[:,:,3]]
    transformed = self.transforms(image=image, masks=masks)
    transformed_image = transformed['image']
    transformed_masks = transformed['masks']
    transformed_masks = np.stack(transformed_masks,  axis=-1)
    return transformed_image, transformed_masks

  def tf_augment(self, image, mask):
    im_shape = image.shape
    msk_shape = mask.shape
    #print(f"Inside tf_aug_fn: image.shape = {image.shape}, mask.shape = {mask.shape}")
    [aug_image, aug_mask] = tf.numpy_function(self.aug_fn, [image, mask], [tf.float32, tf.float32])
    aug_image.set_shape(im_shape)
    aug_mask.set_shape(msk_shape)
    return aug_image, aug_mask

#CHECKS. Visualize tf.data.dataset elements
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask class 0', 'True Mask class 1', 'True Mask class 2', 'True Mask class 3']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    if i == 0:
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    else:
      plt.imshow(display_list[i], cmap='gray')
    plt.axis('off')
  plt.show()


def plot_n_elements(ds,n):
  for images, masks in ds.take(n):
    sample_image, sample_mask = images[0], masks[0]
    #print('mask shape',masks.shape)
    display([sample_image, sample_mask[:,:,0], sample_mask[:,:,1], sample_mask[:,:,2], sample_mask[:,:,3]])
  return

#plot_n_elements(data,2)



class parser_imseg_one_class():
  def __init__(self, input_path_image, input_path_mask, input_img_size, input_classes_cnt, input_class_num):
      self.path_image = input_path_image
      self.path_mask = input_path_mask
      self.img_size = input_img_size
      self.classes_cnt = input_classes_cnt
      self.class_num = input_class_num


  def convert_mask_to_confidence_tensor(self, mask):
    output_shape = mask.shape
    output = np.zeros(shape=output_shape)
    output[mask == self.class_num] = 1
    return np.ndarray.astype(output, np.float32)

  def create_masks(self, mask):
    msk_shape = mask.shape
    mask = tf.numpy_function(convert_mask_to_confidence_tensor, [mask], np.float32)
    mask.set_shape(msk_shape)
    return mask

  def parse_image(self, filename):
    image = tf.io.read_file(self.path_image + filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, [self.img_size, self.img_size])
    return image

  def parse_mask(self, filename):
    mask = tf.io.read_file(self.path_mask + filename)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [self.img_size, self.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = self.create_masks(mask)
    return mask

