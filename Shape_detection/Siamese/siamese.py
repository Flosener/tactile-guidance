import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the pre-trained Siamese model
model = load_model('siamese_mnist_best.h5')

def preprocess_image(image_path):
    """Load an image, convert to grayscale, resize, normalize and expand dimensions."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = cv2.resize(img, (28, 28))                     # Resize to 28x28 pixels
    img = img.astype('float32') / 255.0                 # Normalize pixel values
    img = np.expand_dims(img, axis=-1)                  # Add channel dimension
    return img

def make_prediction(img1, img2, model):
    """Use the Siamese network to predict if two images are similar."""
    img1 = np.expand_dims(img1, axis=0)  # Add batch dimension
    img2 = np.expand_dims(img2, axis=0)  # Add batch dimension
    prediction = model.predict([img1, img2])
    predicted_label = 'Same' if prediction[0][0] > 0.5 else 'Different'
    return predicted_label

# Image paths
image_path1 = 'path/to/your/image1.jpg'
image_path2 = 'path/to/your/image2.jpg'

# Preprocess the images
img1 = preprocess_image(image_path1)
img2 = preprocess_image(image_path2)

# Predict if the images are the same or different
result = make_prediction(img1, img2, model)
print(f'The predicted label for the images is: {result}')