from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from tensorflow import keras
import PIL
import glob
from sklearn.model_selection import train_test_split
import shutil
import os

img_width, img_height = 28, 28
input_shape = (img_width, img_height, 1)

if __name__ == '__main__':
	model = keras.models.load_model('model.h5')
	os.chdir(os.getcwd() + '/test')
	print( os.getcwd() )

	test_data = []
	test_labels = []
	for file in glob.glob("*.png"):
	    img = PIL.Image.open(file).convert('L')
	    imgarr = np.array(img)
	    for i in range( len(imgarr) ):
	        for j in range( len(imgarr[i]) ):
	            imgarr[i][j] = 255 - imgarr[i][j]
	    test_data.append(imgarr)
	    test_labels.append( int(file.split('.')[0][0]) )
	test_labels = np.array(test_labels)
	test_data = np.array(test_data)

	test_data = test_data / 255

	test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
	scores = model.predict(test_data)

	for i in scores:
		print(np.argmax(i), end=' ')
	print()
	print(test_labels)
