import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
import numpy as np

train_dir = 'data/train'
test_dir = 'data/test'
img_height, img_width = 128, 128
batch_size = 16
epochs = 30

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)

# Base model function for creating fresh models
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in base_model.layers[:-10]:  # tune only the last 10 layers
        layer.trainable = False # freeze the base model

    model = Sequential([
        base_model,
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.6),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
checkpoint_callback = lambda name: ModelCheckpoint(f'{name}.keras', monitor='val_loss', save_best_only=True)

# Default model
print("\nTraining default m")
default_model = create_model()
default_train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    zoom_range=0.4,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.1,
    brightness_range=[1, 1], # no brightness adjustment
    shear_range=0.3,
    fill_mode='nearest'
)
default_train_generator = default_train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)
default_model.fit(default_train_generator, epochs=epochs, validation_data=test_generator,
                  callbacks=[reduce_lr, early_stopping, checkpoint_callback('default_detector')])
print("Default model saved.")


# brightness-robust model
print("\nTraining Brightness-Robust Model")
brightness_model = create_model()
brightness_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    brightness_range=[1.0, 2.0]
)
brightness_train_generator = brightness_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)
brightness_model.fit(brightness_train_generator, epochs=20, validation_data=test_generator,
                     callbacks=[reduce_lr, early_stopping, checkpoint_callback('brightness_robust_detector')])
print("Brightness-robust model saved.")


# noise-robust model
print("\nTraining noise-robust model")
noise_model = create_model()
def add_noise(image, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0)

noise_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    preprocessing_function=lambda img: add_noise(img, noise_factor=0.1)
)
noise_train_generator = noise_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)
noise_model.fit(noise_train_generator, epochs=20, validation_data=test_generator,
                callbacks=[reduce_lr, early_stopping, checkpoint_callback('noise_robust_detector')])
print("Noise-robust model saved.")

# Low-light model training
print("\nTraining Low-Light Model")
low_light_model = create_model()
darkness_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    brightness_range=[0.1, 0.7]
)
darkness_train_generator = darkness_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)
low_light_model.fit(darkness_train_generator, epochs=20, validation_data=test_generator,
                    callbacks=[reduce_lr, early_stopping, checkpoint_callback('low_light_detector')])
print("Low-light model saved.")
