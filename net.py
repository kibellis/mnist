from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == '__main__':
	train = pd.read_csv("mnist_train.csv")
	Y = train['label']
	X = train.drop(['label'], axis=1)

	# Разделяем на обучающую выборку и выборку валидации
	x_train, x_val, y_train, y_val = train_test_split(X.values, Y.values, test_size=0.18, random_state=1000)
	print(x_train.shape, y_train.shape)
	print(x_val.shape, y_val.shape)

	batch_size = 128
	num_classes = 10
	epochs = 5

	# размерность картинки
	img_rows, img_cols = 28, 28

	# преобразование обучающей выборки
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_train = x_train.astype('float32')
	x_train /= 255

	# преобразование выборки валидации
	x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
	x_val = x_val.astype('float32')
	x_val /= 255

	input_shape = (img_rows, img_cols, 1)

	# преобразование отклика в 10  перменных
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
	model.summary()

	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
	accuracy = model.evaluate(x_val, y_val, verbose=0)
	print('Test score:', accuracy[0])
	print('Test accuracy:', accuracy[1])

	model.save('model.h5')
