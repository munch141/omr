import PIL
from PIL import Image
import numpy as np
import os


def clases():
    with open('list.txt') as f:
        clases = f.readlines()
    clases = [x.strip() for x in clases]
    return clases


def normalizar(imgname, width, height):
    img = Image.open(imgname).convert('L')
    img = img.resize((width, height), PIL.Image.ANTIALIAS)
    img = np.array(list(img.getdata()))/255.0
    return img


def load_training_data():
    folders = clases()
    train_data = []
    train_labels = []
    for i, folder in enumerate(folders):
        directory = 'training/'+folder+'/'
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                train_data.append(normalizar(directory+filename, 28, 28))
                train_labels.append(i)
    return train_data, train_labels


def load_testing_data():
    folders = clases()
    test_data = []
    test_labels = []
    for i, folder in enumerate(folders):
        directory = 'testing/'+folder+'/'
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                test_data.append(normalizar(directory+filename, 28, 28))
                test_labels.append(i)
    return test_data, test_labels


def load_validation_data():
    folders = clases()
    validation_data = []
    validation_labels = []
    for i, folder in enumerate(folders):
        directory = 'validation/'+folder+'/'
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                validation_data.append(normalizar(directory+filename, 28, 28))
                validation_labels.append(i)
    return validation_data, validation_labels
