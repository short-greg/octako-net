# Takonet: For faster machine learning research

*Takonet* is a framework using Pytorch for fast machine learning R&D. It relieves the pain of developing software for research through modularization and event-driven design. Making it easier to engineer learning machines and the process of making them learn. 

## On the surface

The word tako in Takonet comes from the Japanese word for octopus, an almost alien-like animal with alien-like, distributed, flexible intelligence. Takonet creates such a "tako-like", flexible framework for building learning machines and doing machine learning research.

## Diving In

If you've done some machine learning with neural networks, you may have found yourself wanting to make some alterations to the network structure, the learning algorithm, or just add in another component which displays the weights. So what do you do? In the best case, you copy one file to another, make some modifications and train again. This process is prone to error and is inflexible. 

Takonet solves these ails.

1. The headache of building networks. Networks can be flexibly designed and queried. 
2. The aggravation of changing hyperparameters. Networks can easily be rebuilt with new parameters.
3. The pains of altering the training process. The event-driven design modularizes all components in the learning process so new components can easily be added in. 

## Diving Deeper

Takonet is divided into two main packages: Machinery and Teaching.

### Machinery: 
For designing and building flexible learning machines

* Network: Used to design complex networks in a manner similar to Keras but without the magic.
* Builders: For building the operations in a network. 
* Assemblers: For building networks. They separate the hyperparameters to tune, from the logic of building the network.
* Learners: Learning machines. Bridge the learning process with the network

<Provide Example>

### Teaching: 
For designing and building a flexible learning process

* Dojo: Organizes the parameters for teaching and builds courses.
* Course: For running the whole training process.
* Teachers: Train, test, apply the learner.
* Observers: Respond to teaching events. Used for 

<Provide Example>
