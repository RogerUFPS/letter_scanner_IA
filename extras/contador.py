import tensorflow as tf
import tensorflow_datasets as tfds

train_dataset = tfds.load('emnist/letters', split='train', as_supervised=True)
print(f"Número total de imágenes de entrenamiento: {tf.data.experimental.cardinality(train_dataset).numpy()}")

