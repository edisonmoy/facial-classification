import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
import os

img_path = 'downloads'

def get_size_statistics():
    heights = []
    widths = []
    img_count = 0
    for directory in os.listdir(img_path):
        for img in directory:
            path = os.path.join(directory, img)
            if "DS_Store" not in path:
                data = np.array(Image.open(path))
                heights.append(data.shape[0])
                widths.append(data.shape[1])
                img_count += 1
        avg_height = sum(heights) / len(heights)
        avg_width = sum(widths) / len(widths)
        print("Average Height: " + str(avg_height))
        print("Max Height: " + str(max(heights)))
        print("Min Height: " + str(min(heights)))
        print('\n')
        print("Average Width: " + str(avg_width))
        print("Max Width: " + str(max(widths)))
        print("Min Width: " + str(min(widths)))

get_size_statistics()

def label_img(directory):
    if directory == 'chinese_people': return np.array([1, 0])
    elif word_label == 'ghanaian_people' : return np.array([0, 1])

IMG_SIZE = 300
    
def load_data():
    train_data = []
    for directory in os.listdir(img_path):
        for img in directory:
            label = label_img(directory)
            path = os.path.join(directory, img)
            if "DS_Store" not in path:
                img = Image.open(path)
                img = img.convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
                train_data.append([np.array(img), label])
    shuffle(train_data)
    return train_data

train_data = load_training_data()
plt.imshow(train_data[13][0], cmap = 'gist_gray')


trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train_data])


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


model.fit(trainImages, trainLabels, batch_size = 50, epochs = 5, verbose = 1)



# Test on Test Set
TEST_DIR = './test'
def load_test_data():
    test_data = []
    for img in os.listdir(TEST_DIR):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            test_data.append([np.array(img), label])
    shuffle(test_data)
    return test_data


test_data = load_test_data()    
plt.imshow(test_data[10][0], cmap = 'gist_gray')
        
