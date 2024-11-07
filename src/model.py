import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2

train_dir = 'data/train'
test_dir = 'data/test'

img_height, img_width = 128, 128
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    zoom_range=0.4,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    brightness_range=[0.3, 1.7],
    shear_range=0.3,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

for layer in base_model.layers[:-10]:  # tune only the last 10 layers
    layer.trainable = False

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

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_traffic_light_detector.keras', monitor='val_loss', save_best_only=True)

epochs = 50
model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[reduce_lr, early_stopping, checkpoint])
model.save('final_traffic_light_detector.keras')
print("Model saved")

brightness_finetune_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    brightness_range=[0.3, 2.0]
)

brightness_train_generator = brightness_finetune_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

print("\n fine-tuning with brightness variations.")
model.fit(brightness_train_generator, epochs=10, validation_data=test_generator, callbacks=[reduce_lr, early_stopping])
model.save('brightness_finetuned_traffic_light_detector.keras')
print("Model saved.")