# -*- coding: utf-8 -*-


from keras.models import load_model
from skimage.io import imread
from efficientnet.keras import center_crop_and_resize, preprocess_input
import numpy as np
from scipy.spatial.distance import euclidean

def prepare_models(path_to_model='model.h5'):
    '''Loads and prepares the model.
    path_to_model: File Path that leads to the h5 file of the model.
    This parameter is only needed if the model and this script are saved in different folders.'''
    global model, base_model

    # Load Siamese Network
    print('Loading model...')
    model = load_model('model.h5', compile=False)
    print('Done.')
    
    # Get the base model to calculate feature vectors of single images
    base_model = model.layers[2]

def load_and_preprocess_image(path):
    '''loads an image and preprocesses it.
    path: path leading to the image file.
    returns: numpy array with preprocessed image'''

    img = imread(path)
    img = center_crop_and_resize(img, image_size=256)
    return preprocess_input(img)

def predict_vector_distance(img1, img2):
    '''predicts the euclidian vector distance between two cat images.
    img1: preprocessed image with cat 1
    img2: preprocessed image with cat 2
    returns: floating point value of euclidian distance'''
    
    img1 = np.expand_dims(img1, 0)
    img2 = np.expand_dims(img2, 0)
    return model.predict([img1, img2]).item()

def predict_vector_of_single_image(img):
    '''predicts a feature vector of a single image
    img: preprocessed image to calculate the vector from.
    returns: 128-dimensional vector'''

    img = np.expand_dims(img, 0)
    return base_model.predict(img)[0]

def predict_vector_distances_of_batch_of_image_pairs(img_pairs):
    '''predicts the euclidian vector distance between mulltiple image pairs of cats.
    img_pairs: list of two arrays with one image of a pair in each
    returns: array with floating point values of the euclidean distances'''

    return model.predict(img_pairs)

def predict_vectors_of_batch_of_images(imgs):
    '''predicts feature vectors of a batch of images.
    imgs: array with preprocessed images to calculate the vectors from
    returns: array of 128-dimensional vectors'''

    return base_model.predict(imgs)

def sort_cats_by_similarity_to(reference, imgs):
    '''Sorts a list of cats by similarity. This can be used e.g. to sort lost
    reports on websites (or in apps) like www.tasso.net
    reference: list with one or more (preprocessed) images of the cat the other images should be compared with
    imgs: list of cats whereas each cat is a list of one or more images.
    returns: list of sorted cats'''

    # calculate feature vectors for each cat
    vects = [predict_vectors_of_batch_of_images(np.array(cat)) for cat in imgs]

    # calculate feauture vectors for the reference images
    ref_vects = predict_vectors_of_batch_of_images(np.array(reference))

    # calculate the vector distances between each cat images and the reference images
    distances = []
    for cat_vects in vects:
        cat_dists = [np.mean(euclidean(cat_vect, ref_vect)) for cat_vect in cat_vects for ref_vect in ref_vects]
        distances.append(np.mean(cat_dists))

    # sort the cat images by their vector distances  
    indices = np.argsort(distances)
    return np.take_along_axis(np.array(imgs), indices, axis=0)

def sort_cats_by_similarity_with_vectors(reference, imgs, vects):
    '''Sorts a list of cats by similarity using already calculated feauture vectors.
    This can be used e.g. to sort lost reports on websites or in apps like www.tasso.net efficiently.
    reference: List with one or more images of the cat the other images should be compared with
    imgs: list of cats; each cat is a list of one or more images.
    vects: list of cats; each cat is a list of one or more vectors of the images in imgs.
    returns: list of sorted cats.'''

    # calculate feauture vectors for the reference cat
    ref_vects = predict_vectors_of_batch_of_images(np.array(reference))

    # calculate the vector distances between each cat images and the reference images
    distances = []
    for cat_vects in vects:
        cat_dists = [np.mean(euclidean(vect, ref_vect)) for vect in cat_vects for ref_vect in ref_vects]
        distances.append(np.mean(cat_dists))

    # sort the cat images by their vector distances  
    indices = np.argsort(distances)
    return np.take_along_axis(np.array(imgs), indices, axis=0)