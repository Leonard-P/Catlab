# Catlab
AI to identify missing cats

# Our goal
With our Catlab AI we want to help missing cats and their owners. :D
Our goal is to speed up the identification of a found cat by uploading a picture of it in the corresponding app or website which then displays lost reports with similar looking cats. In this way even non-chiped/tattooed cats can be identified within seconds. It is also an improvement in terms of identifying marked cats, as the AI makes it possible to identify it without requiring tools such as chip readers, therefore saving the trip to the shelter or vet.

# Our approach
Our model investigates two pictures of a cat for similarity. We achieve this using a Siamese Network, calculating a feature vector for each of the two entered cat images. The smaller the distance between the vectors, the more similar the cats look. 
The Siamese Network architecture also allows for computing the feature vectors of the photos of the lost reports in advance. Thus, the actual search query only requires a very low processing load. 

# How to use the model
To calculate the feature vector of a single image, use the function calculate_single_image_vector(img) in useful_functions.py or run it from the terminal:
$ python useful_functions.py -s -img


