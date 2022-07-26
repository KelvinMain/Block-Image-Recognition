import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import datetime

photos_dir = "TEMP_DO_NOT_TOUCH/photos after cutting and resizing"
# All photos are expected to be CROPPED before sent to training
# Thus the same is expected before predicting for better result!!!

batch_size = 8
img_height = 189
img_width = 201

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    photos_dir,
    validation_split=0.2,
    subset="training",
    seed=323,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    photos_dir,
    validation_split=0.2,
    subset="validation",
    seed=323,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print(val_ds)
print(train_ds)

normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.

num_classes = len(class_names)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.5, patience=3
)

model.summary()

epochs = 50000
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[callback]
)

# THIS IS CODE TO TEST THE MODEL WRT THE MODEL
"""
TESTING_DIR = "TEMP_DO_NOT_TOUCH\RESULTS\ONEOFEACH_photos_after_cutting_and_resizing"
for filename in os.listdir(TESTING_DIR):
    img = keras.preprocessing.image.load_img(
        TESTING_DIR + "/" + filename, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "{} most likely belongs to {} with a {:.2f} percent confidence."
            .format(filename, class_names[np.argmax(score)], 100 * np.max(score))
    )
"""

ct = str(datetime.datetime.now())
ct = ct.replace(':', '.')

model.save('TEMP_DO_NOT_TOUCH/models/' + ct)
