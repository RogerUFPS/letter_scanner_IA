import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import random

os.makedirs("emnist_imgs", exist_ok=True)

ds = tfds.load('emnist/letters', split='train', as_supervised=True, shuffle_files=True)

sample_list = list(ds.take(100))
random.shuffle(sample_list)

for i, (image, label) in enumerate(sample_list[:5]): 
    plt.imsave(f"./emnist_imgs/{i}_{label.numpy()}.png", image.numpy().squeeze(), cmap='gray')

