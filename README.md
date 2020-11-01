# Catlab
### Artificial Intelligence to identify missing cats

# Who is supposed to read this?
Primarily, this project addresses people who provide a service where cat owners can post missing reports so that poeple who find a cat can identify its owner. If you are such a person and you would like to enhance the user experience with AI, this is the right place to go. <br>
But you may also use the model for your next AI project, like a cat flap with face recognition.

# Our goal
With our Catlab AI, we want to help missing cats and their owners. <br>
Our goal is to speed up the identification of a found cat by uploading a picture of it into your app or website. Our algorithm then displays missing reports with similar looking cats. In this way even non-chipped/tattooed cats can be identified within seconds. It is also an improvement in terms of identifying marked cats, as the AI makes it possible to identify it without requiring tools such as chip readers, therefore saving the trip to the shelter or vet.

# Why use Catlab?
We provide a ready-to-use deep neural network, including the code needed for operation - all for free and under the very permissive MIT-license. From the very beginning, our model was developed with the goals of minimizing computational cost and inference time, without loss of accuracy. To achieve this, we use the brand-new EfficientNet-architecture as backbone and a Siamese Network as overall architecture, which opens up the possibility to run the vast majority of computations already in advance, instead of for every single search request. The computing load of your servers might actually go _down_ as users will have to look through a smaller number of missing reports.

# Our approach
Our model investigates two pictures of a cat for similarity. We achieve this using a Siamese Network, calculating a feature vector for each of the two entered cat images. The smaller the distance between the vectors, the more similar the cats look. 
The Siamese Network architecture also allows for computing the feature vectors of the photos of the missing reports in advance. Thus, the actual search query only requires very low comptutaional cost. 

# How to use the model
The code in catlab_functions.ipynb or catlab_functions.py shows how to use the model to complete tasks like sorting lists of cat images by similarity or calculating feature vectors of cat images. The script also provides ready-to-use functions that are useful when working with the model; the script examples.py demonstrates how to use them.

# Results
* When two images with a vector distance smaller then 0.5 are considered to depict the same cat, our model achieves an accuracy of **85%** on our test set.

In an _unsorted_ list of search results, which contains the missing report of the found cat, the actual "true" search result appears within the first 50% with a probability of 50%. <br>
* In the list sorted by our model, it is located within the first **50%** of the list in over **99.8%** of the cases, and in **66%** of the cases even within the first **5%**. <br>
* On average, the number of search results a user has to look through decreases by over **90%**.

![Histogram](https://github.com/Leonard-P/Catlab/blob/main/histogram.png)
_The right search result appears almost always within the first displayed search results in the sorted list._

# Required Software
* Python 3.7
* Tensorflow 2.1.x
* EfficientNet 1.1.1
