import initial
import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def init(): 
	json_file = open('model_arch.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json,custom_objects={'ZeroPadding': initial.ZeroPadding,'CorrnetCost': initial.CorrnetCost})
	#load woeights into new model
	loaded_model.load_weights("model_wts.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	#loaded_model.comp?ile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = metricsodel.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph