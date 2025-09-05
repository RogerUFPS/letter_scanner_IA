import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./modelo/modelo_reconocimiento_letras.h5')


def predecir_letra(ruta_imagen):
    img = tf.keras.preprocessing.image.load_img(
        ruta_imagen,
        color_mode='grayscale',
        target_size=(28, 28)
    )
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    img_array = tf.transpose(img_array, perm=[1, 0, 2])
    img_array = tf.image.rot90(img_array, k=3)
    img_array = img_array / 255.0
    
    img_array = tf.reshape(img_array, (1, 28, 28, 1))

    print(img_array.shape)

    plt.imshow(img_array[0, :, :, 0], cmap='gray') 
    plt.savefig("output.png")
    plt.title("Imagen preprocesada para el modelo")
    
    prediccion = model.predict(img_array)
    letra_predicha = chr(65 + np.argmax(prediccion))  # A=65 en ASCII

    return letra_predicha


letra = predecir_letra('./test/r.png')
print(f'La letra es: {letra}')

