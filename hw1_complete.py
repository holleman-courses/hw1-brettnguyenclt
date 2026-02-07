#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

from keras import Input
from keras.models import Model

# print(f"TensorFlow Version: {tf.__version__}")
# print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128),
    layers.LeakyReLU(alpha=0.1),
    layers.Dense(128),
    layers.LeakyReLU(alpha=0.1),
    layers.Dense(128),
    layers.LeakyReLU(alpha=0.1),
    layers.Dense(10)  # logits, no activation
  ])
    
    # Compile the model
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )
  return model

def build_model2():
  model = Sequential([
    # 1st Conv2D + BatchNorm
    layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    
    # 2nd Conv2D + BatchNorm
    layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    # 3rd Conv2D + BatchNorm (128 filters)
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    # 4th Conv2D + BatchNorm
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    # 5th Conv2D + BatchNorm
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    # 6th Conv2D + BatchNorm
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    # Flatten + Dense
    layers.Flatten(),
    layers.Dense(10)  # logits, no activation
  ])
    
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )
  return model

def build_model3():
  inputs = Input(shape=(32,32,3))

  # ALL layers are depthwise-separable now
  x = layers.SeparableConv2D(32, (3,3), strides=2, padding='same', activation='relu')(inputs)
  x = layers.BatchNormalization()(x)

  x = layers.SeparableConv2D(64, (3,3), strides=2, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  # Four more separable conv layers
  for _ in range(4):
      x = layers.SeparableConv2D(128, (3,3), padding='same', activation='relu')(x)
      x = layers.BatchNormalization()(x)

  x = layers.Flatten()(x)
  outputs = layers.Dense(10)(x)  # logits

  model = Model(inputs=inputs, outputs=outputs)

  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )

  print("Model3 total parameters:", model.count_params())
  return model

def build_model50k():
  inputs = Input(shape=(32,32,3))

  # First layer: normal Conv2D
  x = layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu')(inputs)
  x = layers.BatchNormalization()(x)

  # Depthwise-separable conv layers
  x = layers.SeparableConv2D(32, (3,3), strides=2, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.SeparableConv2D(48, (3,3), strides=1, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.SeparableConv2D(64, (3,3), strides=1, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  # Global pooling to keep params low
  x = layers.GlobalAveragePooling2D()(x)

  outputs = layers.Dense(10)(x)  # logits

  model = Model(inputs=inputs, outputs=outputs)

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  print("Total parameters:", model.count_params())
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  from sklearn.model_selection import train_test_split

  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  train_images = train_images / 255.0
  test_images  = test_images  / 255.0

  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()

  # Split training set into train + validation
  train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42
  )

  ########################################
  ## Build and train model 1
  
  model1 = build_model1()
  history1 = model1.fit(
  train_images, train_labels,
  epochs=30,
  batch_size=64,
  validation_data=(val_images, val_labels)
  )
  test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=0)
  print("Test Accuracy:", test_acc)

  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)

  model2 = build_model2()
  history2 = model2.fit(
    train_images, train_labels,
    epochs=30,
    batch_size=64,
    validation_data=(val_images, val_labels)
  )

  # Evaluate on test set
  test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose=0)
  print("Model2 Training Accuracy:", history2.history['accuracy'][-1])
  print("Model2 Validation Accuracy:", history2.history['val_accuracy'][-1])
  print("Model2 Test Accuracy:", test_acc2)

  ########################################
  ## Load your test image and classify it
  from keras.utils import load_img

  # Replace './test_image_airplane.png' if your file extension is .jpg
  test_img = np.array(load_img(
    './test_image_airplane.png',
    grayscale=False,
    color_mode='rgb',
    target_size=(32,32)
  ))

  # Add batch dimension and normalize
  test_img = test_img / 255.0
  test_img = np.expand_dims(test_img, axis=0)  # shape (1,32,32,3)

  # Get predictions
  logits = model2.predict(test_img)
  pred_class = np.argmax(logits, axis=1)[0]
  

  # Map index to CIFAR-10 class
  cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  print("Predicted class for test image:", cifar10_classes[pred_class])
  
  
  ### Repeat for model 3 and your best sub-50k params model
  
  model3 = build_model3()
  history3 = model3.fit(
    train_images, train_labels,
    epochs=30,
    batch_size=64,
    validation_data=(val_images, val_labels)
  )

  # Evaluate on test set
  test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=0)
  print("Model3 Training Accuracy:", history3.history['accuracy'][-1])
  print("Model3 Validation Accuracy:", history3.history['val_accuracy'][-1])
  print("Model3 Test Accuracy:", test_acc3)

  ########################################
  ## Build and train sub-50k parameter model
  model50k = build_model50k()
  history50k = model50k.fit(
    train_images, train_labels,
    epochs=30,
    batch_size=64,
    validation_data=(val_images, val_labels)
  )

  # Evaluate on test set
  test_loss50k, test_acc50k = model50k.evaluate(test_images, test_labels, verbose=0)
  print("Best Sub-50k Model Training Accuracy:", history50k.history['accuracy'][-1])
  print("Best Sub-50k Model Validation Accuracy:", history50k.history['val_accuracy'][-1])
  print("Best Sub-50k Model Test Accuracy:", test_acc50k)