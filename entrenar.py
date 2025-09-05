import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

train_dataset, test_dataset = tfds.load('emnist/letters', split=['train', 'test'],as_supervised=True)

def preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.rot90(image, k=3)
    image = tf.cast(image, tf.float32) / 255.0
    label = label - 1
    return image, label

def augment_data(image, label):
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
    
    image = tf.image.central_crop(image, central_fraction=0.95)
    image = tf.image.resize(image, [28, 28])
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.map(augment_data)  
train_dataset = train_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess)
test_dataset = test_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
   
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

print("Entrenando modelo...")
history = model.fit(train_dataset, 
                    epochs=15,
                    validation_data=test_dataset,
                    callbacks=callbacks,verbose=1)

test_loss, test_accuracy = model.evaluate(test_dataset)

model = tf.keras.models.load_model('./modelo/best_model.h5')
model.save('./modelo/modelo_reconocimiento_letras.h5')
print("Modelo guardado como 'modelo_reconocimiento_letras.h5'")
