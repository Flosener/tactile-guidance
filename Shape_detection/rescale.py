from PIL import Image
import numpy as np
import string

def load_and_preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    image = image.convert('L')
    #print(image)
    #data=np.asarray(image)
    #print(data)
    #image.show()
    image = Image.eval(image, lambda x: 0 if x<255 else 255)
    #image.show()
    image = Image.eval(image, lambda x: 255 - x)

    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # If necessary, reshape the array to match the format of EMNIST images
    # Assuming that you want the array to have a shape of (28, 28, 1)
    image_array = image_array.reshape(28, 28, 1)

    return image_array


def int_labels_to_emnist_format(int_labels):
    """Reshapes a list of integer labels to match the format of EMNIST labels."""
    
    # Reshape the integer labels to match the format of EMNIST labels
    emnist_format_labels = np.array(int_labels)

    return emnist_format_labels

def index_to_letter(index):
    """Convert an index to its corresponding letter (a-z)."""
    
    return chr(index + ord('a'))


if __name__ == '__main__':
    # Example usage
    image_path = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/one.jpg'  # Replace with your image path
    preprocessed_image = load_and_preprocess_image(image_path)

    # You can visualize the preprocessed image using Matplotlib
    import matplotlib.pyplot as plt

    plt.imshow(preprocessed_image[..., 0], cmap='gray')  # Use ... to access the first channel
    plt.axis('off')
    plt.show()