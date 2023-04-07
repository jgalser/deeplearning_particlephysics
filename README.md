# Neural Networs for high-energy physics particle classification
##### *by Josué Irad Galindo De la Serna*



High-energy collisions of subatomic particles result in a cascade of novel particles that must be precisely categorized to comprehend the physics of these collisions. However, correctly classifying particles can be difficult due to the significant number of particles generated in a single collision and the difficulty of measuring their properties. To improve the accuracy and efficiency of particle identification, researchers have employed neural networks, a type of machine learning, to classify these particles accurately. The neural network is designed to predict the particle type based on data features such as momentum, energy, and emitted photoelectrons obtained from the Large Hadron Collider. The ability to correctly identify subatomic particles is crucial in advancing our understanding of fundamental physics, and the breakthrough achievement of over 94% accuracy by the model surpasses current methods for particle identification. This development has the potential to have significant implications for future discoveries in particle physics and other related fields.

![image](https://user-images.githubusercontent.com/115569635/230550522-79610b48-c79c-4685-af27-77e23fd317f1.png)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project_intro">Project Introduction</a>
      <ul>
        <li><a href="#business">Business Understanding</a></li>
        <li><a href="#"> </a></li>
        <li><a href="#built-with">Scope</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

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

For the means of this project, data was obtained from a software which simulated the responses obtained from a *silicon detector* and a *photomultipier* ater a highly energetic collision between a proton and an electron, also known as **electron-proton inelastic scattering**. 

The geometry of the simulated collision and the detector is showed in the picture below:
![image](https://user-images.githubusercontent.com/115569635/230621276-b8e97fe0-635a-47e4-a4e2-d45320df1aa1.png)


It is a dataset generatied by a simulation software called GEANT from the CERN laboratories. More information about the simulated detectors is found here> https://www.ge.infn.it/geant4/training/ptb_2009/detector_response.pdf

### About the Model 




![image](https://user-images.githubusercontent.com/115569635/230452743-013f56d1-6e78-4602-a767-476cd12c5a3f.png)
