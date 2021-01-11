---
layout: post
title: How do we perform incremental learning of neural networks in MATLAB?
---

In the deep learning domain, a lot of successful cases can be accredited by incremental learning techniques. There are two advantages to incremental learning. Firstly, with incremental learning, we don't need to load a great amount of data into our memory. For each training iteration, we only need to load the corresponding data, which significantly improves the system efficiency. Furthermore, incremental learning can help us to reduce training time. For traditional machine learning algorithms, if there is any new data coming in, we need to retrain our model. However, in terms of neural networks, we only need to perform several iterations of incremental learning based on the predetermined parameters, which can significantly reduce the learning time and save computational resources.
In MATLAB, the way of performing incremental learning is not so intuitive. In this article, I will present a simple way to perform incremental learning.
Firstly, we need to load experimental data. In this article, we will use body fat dataset as our experimental data.

```matlab
[X,T] = bodyfat_dataset;
X=X.';
T=T.';
```

Subsequently, we should define our model. In this article, we only use a neural network with one hidden layer. 

```matlab
%% Create a network
layers = [
featureInputLayer(size(X,2))
fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(size(T,2))
regressionLayer];
```

Then we can train our network. We only need to pass data and some hyper-parameters to the training function.

```matlab
%% Training
options = trainingOptions('adam','MaxEpochs',10,'Verbose',true);
net = trainNetwork(X,T,layers,options);
layers=net.Layers;
```

Afterwards, if we find that our model doesn't achieve satisfactory performance, we can call the training function again. By doing so, we can further boost the trained network. In order to validate the effect of incremental learning, we can check the training log. Based on the logs, we can see that the training loss has been further improved.
```matlab
%% Incremental learning
net = trainNetwork(X,T,layers,options);
layers=net.Layers;
```

![](https://i.loli.net/2021/01/11/JU8YGRsSC5wIP2E.png)

Finally, a very important thing that I want to mention is that the training function will only perform incremental learning if we pass a pretrained network to it. In the following segment, we create a new network and pass it into the training function. Based on the training log, it's clear that we cannot achieve similar results if we don't leverage the pretrained results.

```matlab
%% Create a new network
layers = [
featureInputLayer(size(X,2))
fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(size(T,2))
regressionLayer];

net = trainNetwork(X,T,layers,options);
layers=net.Layers;
```

![](https://i.loli.net/2021/01/11/5wk6CLgrETSal7Y.png)
