# Catlab
### AI to identify missing cats

# Who is supposed to read this?
Primarily, this project adresses people who provide a service where cat owners can post lost reports so that poeple who find a cat can identify its owner. If you are such a person and you woud like to enhance the user experience with AI, this is the right place to go. <br>
But you may also use the model for your next AI project, like a cat flap with face recognition. :D

# Our goal
With our Catlab AI, we want to help missing cats and their owners. <br>
Our goal is to speed up the identification of a found cat by uploading a picture of it into your app or website which then displays lost reports with similar looking cats. In this way even non-chipped/tattooed cats can be identified within seconds. It is also an improvement in terms of identifying marked cats, as the AI makes it possible to identify it without requiring tools such as chip readers, therefore saving the trip to the shelter or vet.

# Why use Catlab?
We provide a ready-to-use deep neural network, including the code needed for operation - all for free and under the very permissive MIT-licence. From the very beginning, our model was developed with the goals of minimizing computational cost and inference time, whithout loss of accuracy. To achieve this, we use the brand-new EfficientNet-architecture as backbone and a Siamese Network as overall architecture, which opens up the possibility to run the vast majority of computations already in advance, instead of for every single search request. The computing load of your servers might actually go _down_ as users will have to look through a smaller number of lost reports.

# Our approach
Our model investigates two pictures of a cat for similarity. We achieve this using a Siamese Network, calculating a feature vector for each of the two entered cat images. The smaller the distance between the vectors, the more similar the cats look. 
The Siamese Network architecture also allows for computing the feature vectors of the photos of the lost reports in advance. Thus, the actual search query only requires very low comptutaional cost. 

# How to use the model
## Python
To calculate the feature vector of a single image, use the function calculate_single_image_vector(img) in useful_functions.py or run it from the terminal:<br>
$ python useful_functions.py -s --img

To classify wether two images show the same cat, use the function classify_image_pair(img_pair) in useful_functions.py or run it from the terminal:<br>
$ python useful_functions.py -c --img1 --img2

To calculate the vector distance of two cat images, use the function calculate_vector_distance(img_pair) in useful_functions.py or run it from the terminal:<br>
$ python useful_functions.py -d --img1 --img2

## C / java-script / whatever
To calculate the feature vector of a single image, use the function calculate_single_image_vector(img) in useful_functions.py or run it from the terminal:<br>
$ python useful_functions.py -s --img

To classify wether two images show the same cat, use the function classify_image_pair(img_pair) in useful_functions.py or run it from the terminal:<br>
$ python useful_functions.py -c --img1 --img2

To calculate the vector distance of two cat images, use the function calculate_vector_distance(img_pair) in useful_functions.py or run it from the terminal:<br>
$ python useful_functions.py -d --img1 --img2

# Results
* When two images with a vector distance smaller then 0.5 are considered to depict the same cat, our model achieves an accuracy of **85%** on our test set.

In an _unsorted_ list of search results, which contains the lost report of the found cat, the actual "true" search result appears within the first 50% with a probability of 50%. <br>
* In the list sorted by our model, it is located within the first **10%** of the list in over **99.8%** of the cases, and in **66%** of the cases even within the first **1%**. <br>
* On average, the number of search results a user has to look through decreases by a factor of **55** - a time saving of over **98%**.

![Histogram](https://github.com/Leonard-P/Catlab/blob/main/histogram.png)
_The right search result appears almost always within the first displayed search results._
