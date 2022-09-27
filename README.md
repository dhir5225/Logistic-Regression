# Logistic-Regression

### What is Logistic Regression

Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.

Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems.

In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values (0 or 1).

The curve from the logistic function indicates the likelihood of something such as whether the cells are cancerous or not, a mouse is obese or not based on its weight, etc.

### Linear regression vs logistic regression

Linear regression models are used to identify the relationship between a continuous dependent variable and one or more independent variables. When there is only one independent variable and one dependent variable, it is known as simple linear regression, but as the number of independent variables increases, it is referred to as multiple linear regression. For each type of linear regression, it seeks to plot a line of best fit through a set of data points, which is typically calculated using the least squares method.

Similar to linear regression, logistic regression is also used to estimate the relationship between a dependent variable and one or more independent variables, but it is used to make a prediction about a categorical variable versus a continuous one. A categorical variable can be true or false, yes or no, 1 or 0, et cetera. The unit of measure also differs from linear regression as it produces a probability, but the logit function transforms the S-curve into straight line.  

While both models are used in regression analysis to make predictions about future outcomes, linear regression is typically easier to understand. Linear regression also does not require as large of a sample size as logistic regression needs an adequate sample to represent values across all the response categories. Without a larger, representative sample, the model may not have sufficient statistical power to detect a significant effect.

###bImplement linear equation

Logistic Regression algorithm works by implementing a linear equation with independent or explanatory variables to predict a response value. For example, we consider the example of number of hours studied and probability of passing the exam. Here, number of hours studied is the explanatory variable and it is denoted by x1. Probability of passing the exam is the response or target variable and it is denoted by z.

If we have one explanatory variable (x1) and one response variable (z), then the linear equation would be given mathematically with the following equation-

z = β0 + β1x1  

Here, the coefficients β0 and β1 are the parameters of the model.

If there are multiple explanatory variables, then the above equation can be extended to

z = β0 + β1x1+ β2x2+……..+ βnxn

Here, the coefficients β0, β1, β2 and βn are the parameters of the model.

So, the predicted response value is given by the above equations and is denoted by z.

### Logistic Function (Sigmoid Function):

This predicted response value, denoted by z is then converted into a probability value that lie between 0 and 1. We use the sigmoid function in order to map predicted values to probability values. 

In machine learning, sigmoid function is used to map predictions to probabilities. The sigmoid function has an S shaped curve. It is also called sigmoid curve.

1.The sigmoid function is a mathematical function used to map the predicted values to probabilities.

2.It maps any real value into another value within a range of 0 and 1.

3.The value of the logistic regression must be between 0 and 1, which cannot go beyond this limit, so it forms a curve like the "S" form. The S-form curve is called the Sigmoid function or the logistic function.

4.In logistic regression, we use the concept of the threshold value, which defines the probability of either 0 or 1. Such as values above the threshold value tends to 1, and a value below the threshold values tends to 0.

![image](https://user-images.githubusercontent.com/109084435/192530014-09102e0a-bf21-4a55-85ac-7209d61f0f17.png)

![image](https://user-images.githubusercontent.com/109084435/192530157-0777331d-451d-4b9b-9a60-70cf16cc1ea8.png)

Image-Formula of a sigmoid function 

### Decision boundary


The sigmoid function returns a probability value between 0 and 1. This probability value is then mapped to a discrete class which is either “0” or “1”. In order to map this probability value to a discrete class (pass/fail, yes/no, true/false), we select a threshold value. This threshold value is called Decision boundary. Above this threshold value, we will map the probability values into class 1 and below which we will map values into class 0.

Mathematically, it can be expressed as follows:-

p ≥ 0.5 => class = 1

p < 0.5 => class = 0

Generally, the decision boundary is set to 0.5. So, if the probability value is 0.8 (> 0.5), we will map this observation to class 1. Similarly, if the probability value is 0.2 (< 0.5), we will map this observation to class 0. This is represented in the graph below-

![image](https://user-images.githubusercontent.com/109084435/192536223-8033550f-145b-414e-b5af-08a7b2934686.png)

### Assumptions for Logistic Regression:

1.The dependent variable must be categorical in nature.

2.The independent variable should not have multi-collinearity.

3.It requires the observations to be independent of each other. So, the observations should not come from repeated measurements.

4.Logistic Regression model assumes linearity of independent variables and log odds.

### What Are the Types of Logistic Regression?

There are three main types of logistic regression: binary, multinomial and ordinal. They differ in execution and theory. Binary regression deals with two possible values, essentially: yes or no. Multinomial logistic regression deals with three or more values. And ordinal logistic regression deals with three or more classes in a predetermined order. 

#### 1.Binary logistic regression

Binary logistic regression was mentioned earlier in the case of classifying an object as an animal or not an animal—it’s an either/or solution. There are just two possible outcome answers. This concept is typically represented as a 0 or a 1 in coding. Examples include:

1.Assessing cancer risk (outcomes are high or low).

2.Will a team win tomorrow’s game (outcomes are yes or no).

#### 2.Multinomial logistic regression

Multinomial logistic regression is a model where there are multiple classes that an item can be classified as. There is a set of three or more predefined classes set up prior to running the model. Examples include:

1.Predicting whether a student will go to college, trade school or into the workforce.

2.Does your cat prefer wet food, dry food or human food?

#### 3.Ordinal logistic regression

Ordinal logistic regression is also a model where there are multiple classes that an item can be classified as; however, in this case an ordering of classes is required. Classes do not need to be proportionate. The distance between each class can vary. Examples include:

1.Ranking restaurants on a scale of 0 to 5 stars.

2.Predicting the podium results of an Olympic event.

### Steps in Logistic Regression: 

To implement the Logistic Regression using Python, we will use the same steps as we have done in previous topics of Regression. Below are the steps:

1.Data Pre-processing step

2.Fitting Logistic Regression to the Training set

3.Predicting the test result

4.Test accuracy of the result(Creation of Confusion matrix)

5.Visualizing the test set result.

### Advantages of logistic regression

#### 1.Logistic regression is much easier to implement than other methods, especially in the context of machine learning: 

A machine learning model can be described as a mathematical depiction of a real-world process. The process of setting up a machine learning model requires training and testing the model. Training is the process of finding patterns in the input data, so that the model can map a particular input (say, an image) to some kind of output, like a label. Logistic regression is easier to train and implement as compared to other methods.

#### 2.Logistic regression works well for cases where the dataset is linearly separable: 

A dataset is said to be linearly separable if it is possible to draw a straight line that can separate the two classes of data from each other. Logistic regression is used when your Y variable can take only two values, and  if the data is linearly separable, it is more efficient to classify it into two seperate classes.

#### 3.Logistic regression provides useful insights:

Logistic regression not only gives a measure of how relevant an independent variable is (i.e. the (coefficient size), but also tells us about the direction of the relationship (positive or negative). Two variables are said to have a positive association when an increase in the value of one variable also increases the value of the other variable.

### Disadvantages of logistic regression

#### 1.Logistic regression fails to predict a continuous outcome. 

Let’s consider an example to better understand this limitation. In medical applications, logistic regression cannot be used to predict how high a pneumonia patient’s temperature will rise. This is because the scale of measurement is continuous (logistic regression only works when the dependent or outcome variable is dichotomous).

#### 2.Logistic regression assumes linearity between the predicted (dependent) variable and the predictor (independent) variables.

#### 3.Logistic regression may not be accurate if the sample size is too small.

If the sample size is on the small side, the model produced by logistic regression is based on a smaller number of actual observations. This can result in overfitting. In statistics, overfitting is a  modeling error which occurs when the model is too closely fit to a limited set of data because of a lack of training data. Or, in other words, there is not enough input data available for the model to find patterns in it. In this case, the model is not able to accurately predict the outcomes of a new or future datase
