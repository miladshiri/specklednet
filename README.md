# SpeckleDNet
We introduced a dual-path CNN for ultrasound speckle detection termed SpeckleDNet. The network is generated using Tensorflow 2.0 in Python 3.6.

SpeckleDNet has been already trained on the train dataset and can be used in practice just by loading the network model saved as "specklednet_trained.h5". 

In case of using SpeckleDNet or the dataset, please cite the following study:

"Ultrasound Speckle Detection for Tissue Tracking using a Dual-path Convolutional Neural Network", Milad Shiri – Hamid Behnam – Zahra Alizadeh Sani – Niloofar Nematollahi, 2020

## How to use?
There is a test.py file for a quick use of SpeckleDNet. 

## Is it possible to train the model with different configurations? 
Yes, it is. You can change the parameters in "create_and_train_specklednet.py" file and train the model again even with a different dataset. 

