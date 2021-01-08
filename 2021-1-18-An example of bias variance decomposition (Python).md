---
layout: post
title: An example of bias variance decomposition (Python)
---

Bias variance decomposition is a well-known tool for attributing errors in machine learning. In general, the error of the model for a specific problem is related to the stability of this model and the fitting capability of this model. We call the fitting capability of this model bias and call the stability of this model variance.

Bias variance decomposition provides an effective way to determine the source of poor performance, which may further effectively guide us to design a high-quality machine learning model.

Firstly, we should realize that rather than decompose a single error value of a specific model, the bias-variance decomposition concentrates on decomposing the expected error of a set of machine learning models based on different training data. For example, we can build ten different decision trees with different subsets of training data. Then we can calculate the expected error of this algorithm. However, a single value cannot indicate whether our method performs badly due to worse fitting capability or instability. At this time, bias-variance decomposition provides a powerful tool to determine the root of poor performance.

After introducing the basic idea of bias-variance decomposition. Next, we illustrate this method with a code fragment.

First, we use the Boston house price prediction dataset as our exemplar data, and we split the original data into training and test sets with a ratio of 1:1.


```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

X, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
```

Afterward, we need to define the decomposition function. As shown in the following formula, the error is decomposed into two terms, bias and variance. Concerning bias, we define it as the difference between the expected predicted value and actual value. As for variance, we define it as the expectation of the difference between the expected predicted value and each predicted value. Intuitively, for a specific error term, we can put it down to the fact that the model cannot accurately depict the underlying generation process (high bias) or the model is highly unstable (high variance). It's self-evident that we will have high errors if our model has a high bias. As far as high variance, this case corresponds to a scenario where the model is highly sensitive to the training data.

$$
\begin{array}{l}
\mathbb{E}_{\mathcal{D}}\left[\{y(\mathbf{x} ; \mathcal{D})-h(\mathbf{x})\}^{2}\right]
=\underbrace{\left\{\mathbb{E}_{\mathcal{D}}[y(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2}}_{\text {(bias) }^{2}}+\underbrace{\mathbb{E}_{\mathcal{D}}\left[\left\{y(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[y(\mathbf{x} ; \mathcal{D})]\right\}^{2}\right]}_{\text {variance }}
\end{array}
$$

One pitfall that needs to be pointed out is that bias and variance are just two indicators of a specific machine learning algorithm that represents the fitting capability and stability. In the past, people believed that bias and variance were highly correlated to the model complexity. The more complex the model, the higher the variance will be. However, recent years' research shows that over-parameterized models may lead to low bias and variance simultaneously, which contradicts the common belief. All in all, we cannot roughly equate model complexity with bias and variance.




```python
def bias_variance_decomposition(regr):
    print(regr)
    y_pred = []
    for i in range(200):
        sample_indices = np.arange(x_train.shape[0])
        bootstrap_indices = np.random.choice(sample_indices,
                                             size=sample_indices.shape[0],
                                             replace=True)
        regr.fit(x_train[bootstrap_indices], y_train[bootstrap_indices])
        y_pred.append(regr.predict(x_test))
    y_pred = np.array(y_pred)
    bias = (np.mean(y_pred, axis=0) - y_test) ** 2
    variance = np.mean((y_pred - np.mean(y_pred, axis=0)) ** 2, axis=0)
    error = np.mean((y_pred - y_test) ** 2, axis=0)
    print(np.mean(error), np.mean(bias), np.mean(variance))
```

Finally, we will apply bias-variance decomposition to some classical machine learning algorithms, and we plan to find the root of error for these machine learning algorithms. We selected two classical algorithms, decision tree, and linear regression.
First, after calculating related error items of decision tree and linear regression. We discovered that although the expected errors in the decision tree and linear regression are similar, the majority of errors in linear regression originate from bias, while the majority of errors in the decision tree originate from variance. Based on the observed result, we can guess that there exists a huge margin to reduce the error of the decision tree by reducing the variance. Therefore, we try to reduce the variance by using the ensemble method. By applying the bagging method, we found that the error was reduced. Furthermore, from the result of the decomposed error, we find that the reduced error can be attributed to the reduction of variance rather than bias.

In conclusion, in this article, we present an example to illustrate the bias-variance decomposition in the machine learning field. We found that in the case of similar testing errors, there may still exist a lot of possibilities for the source of that error. Therefore, when we need to enhance the algorithm performance, decomposing error into bias and variance is a sensible way to improve the model performance efficiently.


```python
for regr in [LinearRegression(), DecisionTreeRegressor(), BaggingRegressor(DecisionTreeRegressor(),n_estimators=10)]:
    bias_variance_decomposition(regr)
```

    LinearRegression()
    27.14027220569435 25.308338535726147 1.8319336699682045
    DecisionTreeRegressor()
    27.753457114624506 15.438913619565204 12.314543495059288
    BaggingRegressor(base_estimator=DecisionTreeRegressor())
    19.92063215612648 16.614601149357714 3.3060310067687744

