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
            
            print(f"Prediction: {pred_y}, label: {labels}")
            correct += (pred_y == labels).sum()

            '''
            probabilities = F.softmax(test_output, dim=1)
            #print(probabilities)
            #print(probabilities.shape)

            for number, i in enumerate(pred_y):
                predicted_value = i.item()
                print(predicted_value, 'probability:', probabilities[number][predicted_value]*100,'%') '''
                           
    #accuracy = correct / 400 # Our digits test data has 40,000 images
    accuracy = 100 * (correct/len(pred_y))
    print('Test Data Accuracy: {0:.2f}'.format(accuracy))
    return accuracy


def load_images(path, targets):

    images = []

    for target in targets:
        image = load_and_preprocess_image(path + target + '.jpg')
        images.append(image)
    test_images = np.stack(images)

    return test_images

if __name__ == '__main__':
    # Load EMNIST training dataset
    #test_images, test_labels = extract_test_samples('digits')

    participant_ID = input("Enter Participant ID: ")

    image_path = (f'D:/WWU/M8 - Master Thesis/Project/Code/Result/{participant_ID}/')
    #image_path = 'C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/Images/'
    targets = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7', 'a_8', 'a_9']
    test_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    test_images, test_labels = load_images(image_path, targets), int_labels_to_emnist_format(test_labels)

    #test_images = torch.tensor((test_images/255-0.5).reshape(40000, 1, 28, 28))
    test_images = torch.tensor((test_images/255-0.5).reshape(10, 1, 28, 28))
    test_data = list(zip(test_images.float(), test_labels.astype('int64')))

    # Load and test CNN
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=min(10000,len(test_labels)), shuffle=False)
    cnn = torch.load('torch_emnistcnn_checkpoint.pt', map_location=torch.device("cpu"))
    #cnn.cuda()
    test(cnn)

    targets = ['b_0','b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7', 'b_8', 'b_9']
    test_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    test_images, test_labels = load_images(image_path, targets), int_labels_to_emnist_format(test_labels)

    #test_images = torch.tensor((test_images/255-0.5).reshape(40000, 1, 28, 28))
    test_images = torch.tensor((test_images/255-0.5).reshape(10, 1, 28, 28))
    test_data = list(zip(test_images.float(), test_labels.astype('int64')))

    # Load and test CNN
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=min(10000,len(test_labels)), shuffle=False)
    cnn = torch.load('torch_emnistcnn_checkpoint.pt', map_location=torch.device("cpu"))
    #cnn.cuda()
    test(cnn)   
