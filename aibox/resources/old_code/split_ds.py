### Python script to split image dataset into Train, Validation, and Test folders.

# Randomly splits images to 90% train, 5% validation, and 5% test, and moves them to their respective folders.

# library
from pathlib import Path
import random
import os
import csv
import shutil

# same set of random numbers are generated each time
random.seed(42)

def split_dataset(train_percent=0.9, val_percent=0.05):

    # current working directory
    pth = str(Path.cwd())

    # Path needs to be Absolute path
    image_path = pth + '/images'
    train_path = pth + '/train'
    val_path = pth + '/val'
    test_path = pth + '/test'
    csv_path = pth + '/fruithands.csv'

    # Calculate the rows in the csv file (# indiv bbox's)
    csvfile = 'fruithands.csv'

    with open(csvfile) as f:
        numboxes = sum(1 for line in f)

    with open(csvfile,'r') as file:
        reader = csv.reader(file)

        apple = 0
        banana = 0
        orange = 0
        lefthand = 0
        righthand = 0

        for row in reader:
            if row[3] == 'apple':
                apple += 1

            elif row[3] == 'banana':
                banana +=1

            elif row[3] == 'orange':
                orange +=1

            elif row[3] == 'left_hand':
                lefthand +=1

            elif row[3] == 'right_hand':
                righthand +=1

    train_num = int(numboxes*train_percent)
    val_num = int(numboxes*val_percent)
    test_num = numboxes - train_num - val_num

    print('Dataset Information:')
    print('Total Boxes: %d' % numboxes)
    print('Apple Boxes: %d' % apple)
    print('Banana Boxes: %d' % banana)
    print('Orange Boxes: %d' % orange)
    print('Total Hand Boxes: %d, with %d Right and %d Left' % (lefthand+righthand,righthand,lefthand))
    #print('Left Hand: %d' % lefthand)
    #print('Right Hand: %d' % righthand)

    print('Splits Information')
    print('Training Boxes: %d' % train_num)
    print('Validation Boxes: %d' % val_num)
    print('Test Boxes: %d' % test_num)

    # Select 90% of files randomly and move them to train folder

    with open(csvfile) as f:
        reader = csv.reader(f)
        next(reader) # skip the header
        data = [row for row in reader]

    for i in range(train_num):

        chosen_row = random.choice(data)
        img_name = chosen_row[0]

        new_csv = 'train_labels.csv'
        new_csv_path = train_path + '/' + new_csv

        # check if the CSV file exists and if it's empty
        if os.path.isfile(new_csv_path):
            # if the file exists append the new rows without the header
            with open(new_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(chosen_row)
                #print('added row')
        else:
            # if the file does not exist or is empty, write the header and the new rows

            with open(csvfile, mode='r', newline='') as sf, open(new_csv_path, mode='w', newline='') as df:
                reader = csv.reader(sf)
                writer = csv.writer(df)
                writer.writerow(next(reader))
                writer.writerow(chosen_row)
                #print('created file and added row')

        # move the image to train_path folder
        if img_name != 'filename':
            original_path = image_path + '/' + img_name
            new_path = train_path + '/' + img_name
            shutil.copy(original_path, new_path)

        # remove the rand_img from file_list
        data.remove(chosen_row)

    print("After creation of train ds, data is of size" + str(len(data)))

    # Select 5% of remaining files and move them to validation folder

    for i in range(val_num):

        chosen_row = random.choice(data)
        img_name = chosen_row[0]

        new_csv = 'val_labels.csv'
        new_csv_path = val_path + '/' + new_csv

        # check if the CSV file exists and if it's empty
        if os.path.isfile(new_csv_path):
            # if the file exists append the new rows without the header
            with open(new_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(chosen_row)
                #print('added row')
        else:
            # if the file does not exist or is empty, write the header and the new rows

            with open(csvfile, mode='r', newline='') as sf, open(new_csv_path, mode='w', newline='') as df:
                reader = csv.reader(sf)
                writer = csv.writer(df)
                writer.writerow(next(reader))
                writer.writerow(chosen_row)
                #print('created file and added row')

        # move the image to train_path folder
        if img_name != 'filename':
            original_path = image_path + '/' + img_name
            new_path = val_path + '/' + img_name
            shutil.copy(original_path, new_path)

        # remove the rand_img from file_list
        data.remove(chosen_row)
    print("After creation of val ds, data is of size" + str(len(data)))

    # Move remaining files to test folder and divide the remaining bounding boxes between training and validation sets (else they don't get used at all, wasted data)

    whichfile = 'val'

    for i in range(test_num-1):
        chosen_row = random.choice(data)
        img_name = chosen_row[0]

        if whichfile == 'val':
            new_csv = 'val_labels.csv'
            new_csv_path = val_path + '/' + new_csv

            # check if the CSV file exists and if it's empty
            if os.path.isfile(new_csv_path):
                # if the file exists append the new rows without the header
                with open(new_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(chosen_row)
            whichfile = 'train'

        elif whichfile == 'train':
            new_csv = 'train_labels.csv'
            new_csv_path = train_path + '/' + new_csv

            # check if the CSV file exists and if it's empty
            if os.path.isfile(new_csv_path):
                # if the file exists append the new rows without the header
                with open(new_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(chosen_row)
            whichfile = 'val'

        original_path = image_path + '/' + img_name

        if whichfile == 'val':
            new_path = val_path + '/' + img_name
            shutil.copy(original_path, new_path)
            print('image to val')
        if whichfile == 'train':
            new_path = train_path + '/' + img_name
            shutil.copy(original_path, new_path)
            print('image to train')

        test_new_path = test_path + '/' + img_name

        shutil.copy(original_path, test_new_path)

        data.remove(chosen_row)

    print("After creation of test ds, data is of size" + str(len(data)))

split_dataset()