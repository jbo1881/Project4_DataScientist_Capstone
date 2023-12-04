# Project 4: Used Cars Price Prediction

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Motivation](#project-motivation)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

## Overview
This project involves predicting the prices of used cars based on several features such as manufacturer, model, year, mileage, transmission type, drivetrain, fuel type, MPG (Miles Per Gallon), previous accidents, and price reduction from the initial price. The prediction model utilizes a Random Forest Regression algorithm to estimate the price of a used car.

The data comprises information from 762,091 used cars gathered from cars.com, collected in April 2023.


## Installation
### Running the Prediction Interface
1. Ensure Jupyter Notebook is installed.
2. Open the notebook `Cars.ipynb` containing the prediction interface.
3. Download `cars.csv` from `cars.zip`.
4. Execute the cells containing the interface code.
5. Follow the prompts to input the car's details and obtain the predicted price.


## Libraries Used

This project utilizes several Python libraries for data handling, visualization, and machine learning:

1. **pandas (`import pandas as pd`):**
   Pandas is a powerful data manipulation and analysis library that provides data structures like DataFrame, essential for organizing and processing data effectively.

2. **numpy (`import numpy as np`):**
   NumPy is a fundamental library for numerical operations in Python. It's extensively used for mathematical and logical operations on arrays.

3. **datetime (`import datetime`):**
   The datetime module from the Python standard library is used to work with dates and times. In this project, it's used to calculate the age of the cars.

4. **matplotlib (`import matplotlib.pyplot as plt`):**
   Matplotlib is a popular plotting library that helps in creating static, animated, and interactive visualizations in Python. It's crucial for understanding data patterns and trends.

5. **seaborn (`import seaborn as sns`):**
   Seaborn is built on top of Matplotlib and provides an aesthetically pleasing and informative statistical graphics. It helps in creating attractive and informative statistical graphics.

6. **scikit-learn (`from sklearn...`):**
   Scikit-learn is a powerful machine learning library that provides various tools and techniques for building and evaluating machine learning models. In this project, we use functions for data preprocessing, model training, and evaluation.

   - `train_test_split`: Splits the dataset into training and testing sets.
   - `RandomForestRegressor`: Implements the Random Forest regression model.
   - `r2_score`: Computes the R-squared (R2) score for model evaluation.
   - `StandardScaler`: Standardizes features by removing the mean and scaling to unit variance.
   - `GridSearchCV` and `RandomizedSearchCV`: Tools for hyperparameter tuning.
   - `randint` from `scipy.stats`: Generates random integers, useful for hyperparameter tuning.

These libraries are essential for handling data, visualizing insights, and building the machine learning model in this project.


## Project Motivation
The motivation behind this project is to develop a robust and accurate model to predict the prices of used cars. This project addresses several important objectives:

1. **Consumer Assistance:**
   Buying or selling a used car can be challenging, and determining a fair price is crucial. This project aims to provide a tool that assists both buyers and sellers in estimating a reasonable market value for used cars.

2. **Market Understanding:**
   Understanding the factors that influence the price of a used car is essential for market players such as dealers, buyers, and sellers. By creating a predictive model, we gain valuable insights into how various features impact the price.

3. **Machine Learning Application:**
   The project provides a practical application of machine learning techniques, particularly regression, in a domain that's relevant to a wide audience. It showcases the power of data-driven approaches in solving real-world problems.

4. **Improving Decision-Making:**
   Accurate price predictions empower buyers to make informed decisions about purchasing a used car and help sellers set reasonable prices. This ultimately contributes to a fairer and more efficient used car market.

5. **Open Source Contribution:**
   By open-sourcing this project, we aim to contribute to the open source community. Developers and data enthusiasts can learn from, build upon, and enhance this project, fostering collaboration and knowledge sharing.


## File Descriptions

The Jupyter Notebook `Cars.ipynb` combines data manipulation and visualization techniques to analyze the dataset `cars.csv`. This notebook provides insights into price predictions of used cars. It utilizes historical data to make predictions for car prices based on details like manufacturer, model, year, mileage, engine specifications, transmission, drivetrain, fuel type, car history (accidents or damage), owner details, seller information, ratings, and prices.

### Code Structure:
### 1. Exploratory Data Analysis (EDA)
- Analyzed the dataset to understand the distribution and relationships between various features.
- Explored common manufacturers, models, and their average prices.
- Investigated correlations between car age, mileage, accidents, and price.

### 2. Machine Learning Model
- Utilized Random Forest Regression for price prediction.
- Performed data preprocessing, including label encoding categorical variables.
- Conducted hyperparameter tuning using RandomizedSearchCV for model optimization.
- Achieved an R-squared (R2) score of approximately 0.92 for model evaluation.

### 3. User Inputs for Price Prediction
- Created an interactive interface using Jupyter widgets to allow users to input specific features for price prediction.
- The interface prompts users to input details such as manufacturer, model, year, mileage, transmission type, drivetrain, fuel type, MPG, previous accidents, and price reduction.
- Upon entering these details, the system predicts the estimated price of the used car.

  
## Results
  
The main findings of the code can be found in the related Medium post: [Used Cars Price Prediction](https://medium.com/@jaume.bogunaurue/used-cars-price-prediction-d9edb0a4319b).


## Licensing, Authors, Acknowledgements
Credit for the data is attributed to Kaggle. For detailed data licensing and additional descriptive information, you can refer to the Kaggle dataset: [Used Cars Dataset](https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset).
