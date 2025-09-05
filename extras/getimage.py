import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

os.makedirs("emnist_imgs", exist_ok=True)

ds = tfds.load('emnist/letters', split='train', as_supervised=True)

for i, (image, label) in enumerate(ds.take(5)): 
    plt.imsave(f"{i}_{label.numpy()}.png", image.numpy().squeeze(), cmap='gray')

