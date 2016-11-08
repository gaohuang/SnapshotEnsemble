# Snapshot Ensemble: Train 1, Get M for Free
This repository contains the code for the paper [Snapshot Ensemble: Train 1, Get M for Free](http://openreview.net/pdf?id=BJYwwY9ll). 


The code is based on fb.resnet.torch by Facebook (https://github.com/facebook/fb.resnet.torch).


##Table of Contents
0. [Introduction](#intro)
0. [Usage](#usage)
0. [Contact](#contact)

##Introduction
Snapshot Ensemble is a method to obtain ensembles of multiple neural network at no additional training cost. This is achieved by letting a single neural network converge into several local minima along its optimization path and save the model parameters. The repeated rapid convergence is realized using multiple learning rate annealing cycles.

<img src="https://cloud.githubusercontent.com/assets/16090466/20042608/2e5e7c2e-a44b-11e6-8c1b-99e2532011bc.png" width="400"><img src="https://cloud.githubusercontent.com/assets/16090466/20042610/3308fbf0-a44b-11e6-9657-d577be3a0b08.png" width="400">

Figure 1: Left: Illustration of SGD optimization with a typical learning rate schedule. The model converges
to a minimum at the end of training. Right: Illustration of Snapshot Ensembling optimization. The model
undergoes several learning rate annealing cycles, converging to and escaping from multiple local minima. We
take a snapshot at each minimum for test time ensembling.

##Usage 
0. Install Torch ResNet (https://github.com/facebook/fb.resnet.torch);
1. Replace ```train.lua``` with the one from this repository;

2. Add the follwoing options to ```opts.lua``` to support learning rate with cosine annealing

  ```cmd:option('-lrShape',    'multistep',   'learning rate annealing function, multistep or cosine')```
  ```cmd:option('-nCycles',    '1',           'number of learning rate annealing cycles')```
  
3. An example command to train a Snapshot Ensemble with ResNet-110 (B = 200 epochs, M = 5 cycles, Initial learning rate alpha = 0.2) on CIFAR-100:

 ```th main.lua -netType resnet -depth 110 -dataset cifar100 -batchSize 64 -nEpochs 200 -lrShape cosine -nCycles 5 -LR 0.2 -save      
 checkpoints/```


##Contact
gh349 at cornell.edu   
Any discussions, suggestions and questions are welcome!


