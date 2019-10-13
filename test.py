from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

img_rows, img_cols = 28, 28
num_classes = 10

if __name__ == '__main__':
	test = pd.read_csv("mnist_test.csv")

	model = keras.models.load_model('model.h5')
	model.summary()
	Y = test['label']
	X = test.drop(['label'], axis=1)
	y_test = Y.values
	x_test = X.values
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	x_test = x_test.astype('float32')
	x_test /= 255
	y_test = keras.utils.to_categorical(y_test, num_classes)


	test_predict = model.predict(x_test)
	y_test = np.array( [ np.argmax(i) for i in y_test ] )
	test_predict = list( map( np.argmax, test_predict ) )

	con_mat = tf.math.confusion_matrix(labels=y_test, predictions=test_predict).numpy()
	con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

	classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	con_mat_df = pd.DataFrame(con_mat_norm,
                             index = classes,
                             columns = classes)
	figure = plt.figure(figsize=(8, 8))
	sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
