import settings
import glob
import os
from PIL import Image
from resizeimage import resizeimage


class AnimalData:

	@staticmethod
	def get_test_files_path():
		return glob.glob(settings.PREPROCESSED_IMG_DIR + '*.jpg')

	@staticmethod
	def create_preprocessed_images():

		# first it needs to clean dir with preprocessed images
		old_files = glob.glob(settings.PREPROCESSED_IMG_DIR + '*.jpg')
		if old_files:
			for f in old_files:
				os.remove(f)

		# resize images to inception_v3 format
		img_width = 299
		img_height = 299
		file_name_root = 'test_'
		file_name_extension = '.jpg'

		filenames = glob.glob(settings.DOWNLOADED_IMG_DIR + '*.jpg')
		for idx, img in enumerate(filenames):
			fd_img = open(img, 'rb')
			img_r = Image.open(fd_img)

			# resize with saving aspect ratio
			preprocessed_image = resizeimage.resize_contain(img_r, [img_width, img_height])
			preprocessed_image.save(settings.PREPROCESSED_IMG_DIR + file_name_root + str(idx) + file_name_extension, preprocessed_image.format)
			fd_img.close()

		print('Test images were successfully prepared.')


