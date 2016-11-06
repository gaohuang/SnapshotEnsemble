# Snapshot Ensemble: Train 1, Get M for Free
This repository contains the code for the paper [Snapshot Ensemble: Train 1, Get M for Free](http://openreview.net/pdf?id=BJYwwY9ll). 


The code is based on fb.resnet.torch by Facebook (https://github.com/facebook/fb.resnet.torch).


##Table of Contents
0. [Usage](#usage)
0. [Contact](#contact)

##Usage 
0. Install Torch ResNet (https://github.com/facebook/fb.resnet.torch);
1. Replace ```train.lua``` with the one from this repository;
2. Add the follwoing options to ```opts.lua``` to support learning rate with cosine annealing
  ```cmd:option('-lrShape',    'multistep',   'learning rate annealing function, multistep or cosine')```
  ```cmd:option('-nCycles',    '1',           'number of learning rate annealing cycles')```
3. An example command to train a Snapshot Ensemble with ResNet-110 (B = 200 epochs, M = 5 cycles, Initial learning rate \alpha = 0.1) on CIFAR-100:
```th main.lua -netType resnet -depth 110 -dataset cifar100 -batchSize 64 -nEpochs 200 -lrShape cosine -nCycles 5 -LR 0.1 -save checkpoints/```


##Contact
gh349 at cornell.edu   
Any discussions, suggestions and questions are welcome!






# SnapshotEnsemble
