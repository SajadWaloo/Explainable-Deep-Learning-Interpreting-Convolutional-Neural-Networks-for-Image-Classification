import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load pre-trained VGG16 model
model = tf.keras.applications.VGG16(weights='imagenet')

# Define a function to generate saliency maps
def generate_saliency_map(image):
    image = preprocess_input(image)
    image = tf.convert_to_tensor(image[np.newaxis, ...])

    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        top_prediction = predictions[:, np.argmax(predictions[0])]

    saliency_map = tape.gradient(top_prediction, image)
    saliency_map = tf.reduce_max(saliency_map, axis=-1)
    saliency_map = np.array(saliency_map)[0]

    return saliency_map

# Load and preprocess an example image
image_path = 'example_image.jpg'  # Replace with your own image path
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)

# Generate saliency map
saliency_map = generate_saliency_map(image)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image / 255.0)
plt.title('Original Image')
plt.axis('off')

# Display the saliency map
plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title('Saliency Map')
plt.axis('off')

# Show the figure
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load pre-trained VGG16 model
model = tf.keras.applications.VGG16(weights='imagenet')

# Define a function to generate saliency maps
def generate_saliency_map(image):
    image = preprocess_input(image)
    image = tf.convert_to_tensor(image[np.newaxis, ...])

    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        top_prediction = predictions[:, np.argmax(predictions[0])]

    saliency_map = tape.gradient(top_prediction, image)
    saliency_map = tf.reduce_max(saliency_map, axis=-1)
    saliency_map = np.array(saliency_map)[0]

    return saliency_map

# Load and preprocess an example image
image_path = 'example_image.jpg'  # Replace with your own image path
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)

# Generate saliency map
saliency_map = generate_saliency_map(image)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image / 255.0)
plt.title('Original Image')
plt.axis('off')

# Display the saliency map
plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title('Saliency Map')
plt.axis('off')

# Show the figure
plt.tight_layout()
plt.show()
