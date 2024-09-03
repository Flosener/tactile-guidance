import torch
from emnist import extract_test_samples
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from rescale import (load_and_preprocess_image, int_labels_to_emnist_format, index_to_letter)
import torch.nn.functional as F

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def test(cnn):
    '''Calculates the accuracy of the CNN on the test data'''
    cnn.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            #images, labels = images.cuda(), labels.cuda()    
            test_output = cnn.forward(images)
            pred_y = torch.max(test_output, 1)[1]

            pred_letters = [index_to_letter(idx) for idx in pred_y]
            label_letters = [index_to_letter(idx) for idx in labels]
            
            print(f"Prediction: {pred_letters}, label: {label_letters}")
            correct += (pred_y == labels).sum()

            '''
            probabilities = F.softmax(test_output, dim=1)
            #print(probabilities)
            #print(probabilities.shape)

            for number, i in enumerate(pred_y):
                predicted_value = i.item()
                print(predicted_value, 'probability:', probabilities[number][predicted_value]*100,'%') '''
                           
    accuracy = correct / 208 # Our letters test data has 20,800 images
    accuracy = 100 * (correct/len(pred_y))
    print('Test Data Accuracy: {0:.2f}'.format(accuracy))
    return accuracy


def load_images(path, targets):

    images = []

    for target in targets:
        image = load_and_preprocess_image(path + target + '.png')
        images.append(image)
    test_images = np.stack(images)

    return test_images

if __name__ == '__main__':
    # Load EMNIST training dataset
    '''test_images, test_labels = extract_test_samples('letters')'''

    image_path = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/' 
    #image_path = 'C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/Images/'

    targets = ['x', 'y', 'f', '0']
    test_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    test_images, test_labels = load_images(image_path, targets), int_labels_to_emnist_format(test_labels)

    #test_images = torch.tensor((test_images/255-0.5).reshape(20800, 1, 28, 28))
    test_images = torch.tensor((test_images/255-0.5).reshape(4, 1, 28, 28))
    test_data = list(zip(test_images.float(), test_labels.astype('int64')))

    # Load and test CNN
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=min(10000,len(test_labels)), shuffle=False)
    cnn = torch.load('torch_emnistcnn_letter_v2.pt', map_location=torch.device("cpu"))
    #cnn.cuda()
    test(cnn)