# -*- coding: utf-8 -*-
import catlab_functions as catlab
import os
from random import randint
import matplotlib.pyplot as plt

catlab.prepare_models(path_to_model='model.h5')

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

# Load some cat images
imgs = []
for i in range(10):
    cat = []
    folder_path = os.path.join('test_images', str(i))
    for img_name in listdir_nohidden(folder_path):
        # cat.append(catlab.load_and_preprocess_image(os.path.join(folder_path, img_name)))
        print(os.path.join(folder_path, img_name))
    imgs.append(cat)
   
reference_imgs = [
    catlab.load_and_preprocess_image(
        os.path.join('test_images', 'reference', img_name)) 
    for img_name in listdir_nohidden(os.path.join('test_images', 'reference'))]

img1 = catlab.load_and_preprocess_image(os.path.join('test_images', 'img1.jpg'))
img2 = catlab.load_and_preprocess_image(os.path.join('test_images', 'img2.jpg'))

# Execute a random demonstration
case = randint(0, 5)

if case == 0:
    plt.imshow(img1)
    print('Feature vector of this image:',
          catlab.predict_vector_of_single_image(img1))
elif case == 1:
    pass
elif case == 2:
    pass
elif case == 3:
    pass
elif case == 4:
    pass
elif case == 5:
    pass
