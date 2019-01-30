from keras.applications import inception_v3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import matplotlib.pyplot as plt


class AnimalClassifier:

	@staticmethod
	def get_model():
		model = inception_v3.InceptionV3(include_top=True, weights='imagenet')
		return model

	def recognize_animals(self, model, arr_images_path):
		for img_path in arr_images_path:
			self.recognize_animal(model, img_path)

	def recognize_animal(self, model, img_path):

		x, img = self._preprocess_input_data(img_path)

		# get the predicted probabilities for each class
		predictions = model.predict(x)

		# convert the probabilities to class labels
		# We will get top 2 predictions which is the default
		labels = decode_predictions(predictions, top=2)

		# convert prediction to string for easier defining simple class
		label_string = (labels[0][0][1] + ' ' + labels[0][1][1]).lower()

		# my simple classes
		if label_string.find('cat') != -1:
			animal = 'cat'
		elif label_string.find('lion') != -1:
			animal = 'lion'
		elif (label_string.find('horse') != -1) or (label_string.find('sorrel') != -1):
			animal = 'horse'
		else:
			animal = 'undefined'

		print('Original = ' + label_string + '\n My class = ' + animal)

		plt.imshow(img)
		plt.title(animal)
		plt.axis('off')
		plt.show()

	@staticmethod
	def _preprocess_input_data(img_path):
		original_img = load_img(img_path, target_size=(299, 299))

		# Convert the PIL array (width, height, channel) into numpy array (height, width, channel)
		numpy_image = img_to_array(original_img)

		# reshape data in terms of batchsize (batchsize, height, width, channels)
		image_batch = np.expand_dims(numpy_image, axis=0)

		# prepare the image for the Inception model
		processed_image = inception_v3.preprocess_input(image_batch.copy())

		return processed_image, original_img
