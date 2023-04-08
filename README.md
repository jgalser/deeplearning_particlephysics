# Neural Networs for high-energy physics particle classification
##### *by Josué Irad Galindo De la Serna*



High-energy collisions of subatomic particles result in a cascade of novel particles that must be precisely categorized to comprehend the physics of these collisions. However, correctly classifying particles can be difficult due to the significant number of particles generated in a single collision and the difficulty of measuring their properties. To improve the accuracy and efficiency of particle identification, researchers have employed neural networks, a type of machine learning, to classify these particles accurately. The neural network is designed to predict the particle type based on data features such as momentum, energy, and emitted photoelectrons obtained from the Large Hadron Collider. The ability to correctly identify subatomic particles is crucial in advancing our understanding of fundamental physics, and the breakthrough achievement of over 94% accuracy by the model surpasses current methods for particle identification. This development has the potential to have significant implications for future discoveries in particle physics and other related fields.

![image](https://user-images.githubusercontent.com/115569635/230550522-79610b48-c79c-4685-af27-77e23fd317f1.png)


## Project Overview

#### Business Understanding

In high-energy particle collisions, subatomic particles are accelerated to nearly the speed of light and made to collide with each other. These collisions produce a shower of new particles that are detected by sophisticated instruments, such as the detectors in the Large Hadron Collider. 

Classifying these particles is a critical step in understanding the physics of these collisions. By accurately identifying the type of particle produced in a collision, researchers can infer information about the properties of the particles and the forces that govern their behavior. However, **accurately classifying particles can be challenging**, as there are often many particles produced in a single collision and their properties can be difficult to measure due to the existence of different sources of noise. This is where machine learning, specifically **neural networks**, can play a key role in improving particle classification accuracy and efficiency. Implementing this solution will help scientist foccus on understanding the fundamental secrets of the universe rather than spending time dealing with an enormous datase. 

There are several traditional methods that have been used for particle identification in high energy physics, including:
- **Ionization measurements**: This method relies on measuring the amount of ionization produced by a particle as it passes through a detector. Different particles produce different amounts of ionization, which can be used to identify and differentiate them.

- **Time-of-flight measurements**: This method relies on measuring the time it takes for a particle to travel a certain distance through a detector. Different particles have different speeds and masses, which can cause them to arrive at the detector at different times. By measuring the time-of-flight, particles can be identified and differentiated.

- **Cherenkov radiation**: This method relies on detecting the Cherenkov radiation produced by a charged particle as it passes through a medium at a speed greater than the speed of light in that medium. Different particles produce different types of Cherenkov radiation, which can be used to identify and differentiate them.

- **Magnetic field measurements**: This method relies on measuring the curvature of a particle's path as it passes through a magnetic field. Different particles with different charges and masses will be deflected by the magnetic field in different ways, which can be used to identify and differentiate them.

While these traditional methods have been successful in identifying and classifying particles, they have limitations when it comes to handling large and complex datasets, dealing with noise and background events, and identifying new particles and phenomena that may not fit into known categories. 
Some of more about the usual methods used for identifiying/classifiying particles are mention in this article: https://www.sciencedirect.com/science/article/abs/pii/0168900294909849  

The neural network created with this project is designed to **accurately classify** subatomic particles in high-energy particle collisions. The data used in the model consists of various **features** such as energy, momentum, emited photoelectrons,among others, extracted from detectors in the Large Hadron Collider. The **target** of the model is to predict the **particle type** based on these features.

The rationale for predicting this target with this data is that accurately identifying particles in high-energy collisions is critical for advancing our understanding of fundamental physics. By using a neural network to classify particles, we can improve the efficiency and accuracy of particle identification, ultimately leading to more precise measurements and a deeper understanding of the universe.

### The Dataset

For the means of this project, data was obtained from a software which simulated the responses obtained from a *silicon detector* and a *photomultipier* ater a highly energetic collision between a proton and an electron, also known as **electron-proton inelastic scattering**. The used sofware is called GEANT. More information about the simulated detectors can be found here: https://www.ge.infn.it/geant4/training/ptb_2009/detector_response.pdf

For step by step guide on the dataset download and setup refer to the *Getting started* section of this doccument. 

The geometry of the simulated collision and the detector is showed in the picture below:
![image](https://user-images.githubusercontent.com/115569635/230621276-b8e97fe0-635a-47e4-a4e2-d45320df1aa1.png)

When the particles collide, a shower of new particles is produced. Each product particle deflects from the center of the collision to a determined direction with specific velocity and energy. 

The simulated detector generates a response based on how a product particle hits the semiconductor panels (the green flower from the picture above) which will give information about the position and momentum of the particle. The particles that cross beyond these, reach the photomultiplier. The photomultiplier works by generating a shower of photoelectrons when hit due to energy transfer from the incident particle, the response will then be the total number of produced signals of a photoelectron. 

#### Quick view 
![image](https://user-images.githubusercontent.com/115569635/230669368-8134672e-1423-47b3-907a-f966c59acb73.png)

#### What are we classifying? (dataset target)
- **id**: Identification numbrer of the particle event, i.e., what is the product particle after the collision (Positron, Kaon, Pion or Proton). This is our target variable. 
  - Positron: Id = 0 
  - Kaon: Id = 1
  - Pion: Id = 2
  - Proton: Id = 3

#### About the features (dataset columns):

- **p**: Momentum of the particle when it reaches the detector (GeV/c).   
- **theta ($\theta$)**: azimutal angle (radians)
- **beta ($\beta$)**: polar angle (radians)
- **nphe ($N_{\gamma}$)**: number of photoelectrons produced in the photomultiplier
- **ein**: energy in 
- **eout**: energy out 

It is a dataset generatied by a simulation software called GEANT from the CERN laboratories. 

The original dataset contained a total of **5,000,000** rows, after cleanning and resampling, around **58,000** rows of data remained. This drastical reduction on the sample data is due to the high class imbalance presented in the original dataset (Class imbalance in machine learning can lead to biased predictions towards the majority class, resulting in poor performance for the minority class. To address this problem, techniques such as oversampling, undersampling, cost-sensitive learning, or ensemble methods can be employed). For further detail in data preparation, refer to the main project notebook.


### About the Model 
#### Baseline
As a **baseline**, a simple neural network with only 4 neurons in the the hiden layer was trained with the original imbalanced dataset. As expected, the performance of the model was highly affected by the difference between the mayority and minority class, turning out in a good classificator for the mayority class but a completely poor model for identifiying events of the minority one. 

Initially, the class distribution of the data looked like this:
![image](https://user-images.githubusercontent.com/115569635/230681500-6a55ff76-c395-442b-bc5b-b416c020063a.png)
And the resulting evaluation metrics turned out as shown bellow
![image](https://user-images.githubusercontent.com/115569635/230681585-730d1931-6930-4e64-878c-149310fdcc97.png)
- The model was uncapable of identifiying any **Positron** event. 
- For the **Pion** class, the predictions from the model had a low rate of true positives, i.e. , it classified more events incorrectly as pions than the ones classified correctly. However, it performed well identifiying true negatives. 
- For the  **Proton** and **Kaon** classes, most of the true labels where classified correctly but there where also a considerable amount of false labels classified as true.

Overall, this baseline naive model was not even close to be functional or to even meet the goal. 

#### Final
The **final** proposed model consist of a simple neural network, with 6 input neurons (as there are a total of 6 features), one hiden layer with 10 neurons (number of neurons in this layer was chosen trough a grid search exproration between 4 and 12 neurons ) and an output layer with 4 neurons (due to having 4 classes). The architechture is described in the diagram below.

![image](https://user-images.githubusercontent.com/115569635/230452743-013f56d1-6e78-4602-a767-476cd12c5a3f.png)

- Loss Function: **Categorical Cross Entropy**
  - Categorical cross-entropy is a commonly used loss function in machine learning for classification problems where the output variable is a categorical variable. The main reason for using categorical cross-entropy is that it measures the difference between the predicted probability distribution and the true probability distribution of the categories. It does this by computing the log-likelihood of the true categories given the predicted probabilities. This loss function is useful because it penalizes heavily the prediction of a low probability for the true category, while rewarding the prediction of high probabilities for the true category.    

- Optimizer: **Adam** (Adaptive Moment Estimation) is a popular optimization algorithm used in machine learning to update the parameters of a model during training. There are several reasons why one might choose to use Adam as an optimizer:

    - Adaptive learning rate: Adam adapts the learning rate for each parameter based on the historical gradients. This means that it can automatically adjust the learning rate for each parameter and can converge faster than other optimization algorithms, especially in high-dimensional problems.

    - Momentum: Adam uses momentum to accelerate the optimization process. Momentum helps the optimizer to continue moving in the same direction, even when the gradients change direction, which helps to avoid getting stuck in local minima.

    - Regularization: Adam has a built-in regularization mechanism that helps to prevent overfitting by penalizing large weights.

    - Easy to use: Adam is easy to use and doesn't require much tuning of hyperparameters.
 
### Model Results 
After balancing the data, cleaning and hyperparameter tunning, this is how predictions performed compared to true labels:

![image](https://user-images.githubusercontent.com/115569635/230684368-ca8979da-d88d-4c0f-9daf-80c84114feb7.png)

As we see, most predictions are condensed in the diagonal of the confusion matrix, this means that most labels where predicted correctly from the test dataset. 
 
####  Evaluation Metrics 
|           |  precision  |  recall       | f1-score     | support     |
|-----------|------------ | ------------- |--------------|------------ |
|positron    |  96%  |   96%  |    96%  |    5887|
|pion        | 93%    |  89%    |  91%    |  5939|
|proton      | 92%    |  94%    |  93%    |  5965|
|kaon        | 97%    |  98%    |  98%    |  5866|

With the new model, the precission and recall for almost every class was above 90%, having an overal accuracy, of 94%. The classes that performed best with this model where the Positron and Kaon, both with precission and recall above 95%, in contrast with the first version, where the model was uncapable of identifiying the first of these.


### Conclusion 
The development of a neural network that classifies subatomic particles produced in high energy particle collisions has many potential applications in the field of particle physics. By accurately classifying the particles produced, the neural network can aid in the discovery of new particles, as well as the measurement of their properties, which could contribute to our understanding of the fundamental laws of the universe. Furthermore, the classification of subatomic particles can have practical applications in the fields of medical imaging and radiation therapy, where the precise measurement and classification of subatomic particles can help to improve treatment planning and outcomes. Overall, the development and implementation of a neural network for subatomic particle classification has significant potential to advance both theoretical and practical applications in various fields.

Some of the applications where this model can come to play can be found in the following links:

- CERN (European Organization for Nuclear Research) - https://home.cern/science/physics

- Particle Physics Applications - https://www.physics.purdue.edu/research/high_energy_physics/applications.php

- Medical Applications of Particle Physics - https://www.iaea.org/topics/radiation-protection/medical-applications-of-ionizing-radiation/particle-therapy

- Brookhaven National Laboratory - Applications of High Energy Physics - https://www.bnl.gov/science/high-energy-physics/applications.php

- Fermilab - Particle Physics for Everyone - https://www.fnal.gov/pub/science/particle-physics/particle-physics-for-everyone/what-is-particle-physics/applications/index.html


 ## Getting Started

There are a couple of steps to follow to run this repository localy. 

#### Running environment
This is a project that runs completely on python 3 Jupyter Notebooks. Be sure to have the needed IDE to run a jupyter notebook.

#### Prerequisites
**Download and install the required libraries**
The very first step is to have the proper python packages installed. These are the ones we will be using:
  * **Numpy** 
  
    Version: 1.19.5

    NumPy is the fundamental package for array computing with Python.

    Home-page: https://www.numpy.org 

    Command: 
    ```sh
    pip3 install numpy==1.19.5
    ```
  * **Pandas** 
  
    Version: 1.1.2

    Powerful data structures for data analysis, time series, and statistics

    Home-page: https://pandas.pydata.org 

    Command: 
    ```sh
    pip3 install pandas==1.1.2
    ```
  * **Seaborn** 
  
    Version: 0.11.2

    Powerful tool for statistical data visualization

    Home-page: https://seaborn.pydata.org

    Command: 
    ```sh
    pip3 install seaborn==0.11.2
    ```
  * **Matplotlib** 
  
    Version: 3.3.4

    Python plotting package

    Home-page: https://matplotlib.org
    
    Command: 
    ```sh
    pip3 install matplotlib==3.3.4
    ```
  
  * **Keras** 
  
    Version: 2.6.0

    TensorFlow Keras.

    Home-page: https://keras.io/
    
    Command: 
    ```sh
    pip3 install keras==2.6.0
    ```
   
  * **Imblearn** 
  
    Version: 0.0

    Toolbox for imbalanced dataset in machine learning.

    Home-page: https://pypi.python.org/pypi/imbalanced-learn/
    
    Command: 
    ```sh
    pip3 install imblearn==0.0
    ```
#### Clone the repo

Fork this repo to make a copy on your github account. Then clone it locally. 

#### Download the dataset 
Once all libraries are properly installed, it is time to **download the dataset** we used for the model.

 
1. As we are extracting the data from Kaggle, to download the dataset you will have to create an account in the Kaggle homepage (https://www.kaggle.com/).

![image](https://user-images.githubusercontent.com/115569635/230698111-07b18682-d323-47de-882a-f4ea5c81f00c.png)

2. After successfully creating an account, access the following link and click **download**: https://www.kaggle.com/datasets/naharrison/particle-identification-from-detector-responses 

![image](https://user-images.githubusercontent.com/115569635/230697992-7462bd20-d55d-454a-846b-7b0ae8585e97.png)

The zip file should have the name **pid-5M.csv.zip**

3. Locate your copy of this repo. Extract the csv inside the folder named **Collision Dataset** and rename the file to *collision_simulations.csv* 

#### NOW EVERYTHING IS SET TO RUN THE MAIN PROJECT NOTEBOOK!

## Repository Exploration Guide

##### 1. Collision Dataset [Folder]
This is where the dataset should be downloaded

##### 2. *Capstone Project Proposal.pdf* [file]
This is a pdf file that contains the original proposal for the project

##### 3. *Main.ipynb* [file]
This is the notebook that contains the whole project, from data exploration to training and evaluation
Link: https://github.com/jgalser/deeplearning_particlephysics/blob/main/Main.ipynb 

##### 4. *NN_for_ParticleClassification_Presentation.ipynb* [file]
This is the pdf that contains the slides for the non technical project presentation.
Link: https://github.com/jgalser/deeplearning_particlephysics/blob/main/NN_for_ParticleClassification_Presentation.pdf

##### 5. *README.md* [file]
This Readme file

