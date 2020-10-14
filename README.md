# Catlab
AI to identify missing cats

# Our goal
With our Catlab AI we want to help missing cats and their owners. :D
Our goal is to speed up the identification of a found cat by uploading a picture of it in the corresponding app or website which then displays lost reports with similar looking cats. In this way even non-chiped/tattooed cats can be identified within seconds. It is also an improvement in terms of identifying marked cats, as the AI makes it possible to identify it without requiring tools such as chip readers, therefore saving the trip to the shelter or vet.

# Why use Catlab?
From th very beginning, our model was developed with the goals of minimizing computational cost and inference time whithout loss of accuracy. To archive this, we use the brand-new EfficientNet-architecture as a backbone and a Siamese Net as overall architecture, which opens up the possibility to run the vast majority of computations already before inference time instead of for every single search request. We provide a ready-to-use deep neural network, including the code needed for inference, erverything with the very permissive MIT-licence

# Our approach
Our model investigates two pictures of a cat for similarity. We achieve this using a Siamese Network, calculating a feature vector for each of the two entered cat images. The smaller the distance between the vectors, the more similar the cats look. 
The Siamese Network architecture also allows for computing the feature vectors of the photos of the lost reports in advance. Thus, the actual search query only requires very low comptutaional cost. 

# How to use the model
## Python
To calculate the feature vector of a single image, use the function calculate_single_image_vector(img) in useful_functions.py or run it from the terminal:
$ python useful_functions.py -s --img

To classify wether two images show the same cat, use the function classify_image_pair(img_pair) in useful_functions.py or run it from the terminal:
$ python useful_functions.py -c --img1 --img2

To calculate the vector distance of two cat images, use the function calculate_vector_distance(img_pair) in useful_functions.py or run it from the terminal:
$ python useful_functions.py -d --img1 --img2

## C / java-script / whatever
To calculate the feature vector of a single image, use the function calculate_single_image_vector(img) in useful_functions.py or run it from the terminal:
$ python useful_functions.py -s --img

To classify wether two images show the same cat, use the function classify_image_pair(img_pair) in useful_functions.py or run it from the terminal:
$ python useful_functions.py -c --img1 --img2

To calculate the vector distance of two cat images, use the function calculate_vector_distance(img_pair) in useful_functions.py or run it from the terminal:
$ python useful_functions.py -d --img1 --img2

# Evaluation
