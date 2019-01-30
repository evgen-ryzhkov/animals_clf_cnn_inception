from scripts.data import AnimalData
from scripts.model import AnimalClassifier


#  --------------
# Step 1
# getting test data
data_o = AnimalData()
# data_o.create_preprocessed_images()
test_files_path = data_o.get_test_files_path()

#  --------------
# Step 2
# building model
clf_o = AnimalClassifier()
clf_model = clf_o.get_model()



#  --------------
# Step 3
# recognize images
# test_img = 1
# clf_o.recognize_animals(clf_model, arr_images_path=test_files_path)
clf_o.recognize_animal(clf_model, img_path=test_files_path[5])
