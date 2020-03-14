from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# Create checkpoint callbacks
checkpoint_path = "checkpoint/model.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch')
earlystopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')

def listDir(directory):
  filelist = os.listdir(directory)
  return [x for x in filelist if not (x.startswith('.'))]

train_data_dir = pathlib.Path('train')
val_data_dir = pathlib.Path('validate')

numClasses = sum(os.path.isdir(os.getcwd() + '/train/' + i) for i in listDir(os.getcwd() + '/train'))
numTrainingImages = 0
numValImages = 0
for root, dirs, files in os.walk('train'):
  numTrainingImages += len(files)

for root, dirs, files in os.walk('validate'):
  numValImages += len(files)

train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=45,
  width_shift_range=0.2,
  height_shift_range=0.2,
  brightness_range=[0.2, 1.0],
  zoom_range=[0.5, 1.0],
  horizontal_flip=True,
  data_format="channels_last"
)

val_datagen = ImageDataGenerator(
  rescale=1./255,
  data_format="channels_last"
)

def make_train_gen():
  return train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(IMG_WIDTH, IMG_HEIGHT),
  color_mode="rgb",
  class_mode="categorical",
  batch_size=BATCH_SIZE
)

def make_val_gen():
  return val_datagen.flow_from_directory(
  val_data_dir,
  target_size=(IMG_WIDTH, IMG_HEIGHT),
  color_mode="rgb",
  class_mode="categorical",
  batch_size=BATCH_SIZE
)

train_ds = tf.data.Dataset.from_generator(
  make_train_gen,
  output_types=(tf.float32, tf.float32),
  output_shapes=([BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3], (BATCH_SIZE, numClasses))
)

val_ds = tf.data.Dataset.from_generator(
  make_val_gen,
  output_types=(tf.float32, tf.float32),
  output_shapes=([BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3], (BATCH_SIZE, numClasses))
)

for image_batch, label_batch in train_ds.take(1):
  pass

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(numClasses)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate, momentum=0.9, decay=1e-6),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['categorical_accuracy'])
model.summary()

print(len(model.trainable_variables))

initial_epochs = 1000
training_steps = numTrainingImages // BATCH_SIZE
validation_steps = numValImages // BATCH_SIZE 

loss0,accuracy0 = model.evaluate(val_ds, steps=validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(
  train_ds,
  epochs=initial_epochs,
  steps_per_epoch=training_steps,
  validation_data=val_ds,
  validation_steps=validation_steps,
  callbacks=[checkpoint_cb, earlystopping_cb]
)

# Summary
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Categorical Accuracy')
plt.plot(val_acc, label='Validation Categorical Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Categorical Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Categorical Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
