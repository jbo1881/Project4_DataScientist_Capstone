# Project 4: Used Cars Price Prediction

## Table of Contents

- [Installation](#installation)
- [Project Motivation](#project-motivation)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

## Installation
The following Python libraries should be installed:
 
- pandas (import pandas as pd): A versatile library used for data manipulation and analysis. It provides data structures like DataFrames that allow you to efficiently work with structured data.
- matplotlib.pyplot (import matplotlib.pyplot as plt): A part of the matplotlib library used for creating various types of visualizations, such as plots and charts.
- sklearn (from sklearn.model_selection import train_test_split, from sklearn.linear_model import LinearRegression, from sklearn.metrics import mean_squared_error): Part of the Scikit-learn library, which provides tools for machine learning and data analysis. It's used here for linear regression modeling and evaluation.
The data was collected for a period of 2014 - 2022 years, divided into trimesters. The prices go by neighbourhoods and districts.

## Project Motivation
In order to complete the Project: Writing a Data Scientist Blog Post from the Data Scientist course of Udacity, the dataset named 'Rent price in Barcelona 2014 - 2022' from Kaggle includes data on price for rent in Barcelona, Spain. 

The dataset was filtered to answer these 3 questions I was interested in:
1. What Are the Most Expensive and Affordable Neighborhoods for Renting in Barcelona by Year?
2. How Have Rental Prices Evolved Over the Years in Different Districts of Barcelona?
3. Are There Seasonal Patterns in Barcelona's Rental Prices Across Trimesters?

## File Descriptions

The Jupyter Notebook `bcn_rent_prices.ipynb` combines data manipulation and visualization techniques to analyze the dataset `Barcelona_rent_price.csv`. This notebook provides insights into rental price trends in Barcelona's neighborhoods and districts. It also utilizes historical data to make predictions for future rental prices.

Code Sections:
1. Analyzing Neighborhood Rental Prices:
   - Loads rental price data into a pandas DataFrame.
   - Filters and focuses on rows with 'average rent (euro/month)' prices.
   - Groups data by year and neighborhood, calculates the mean price, and creates a DataFrame (neigh_rents).
  
2. Visualizing Most and Least Expensive Neighborhoods:
   - Defines functions to find most and least expensive neighborhoods for each year.
   - Iterates through years and stores results in a list.
   - Prepares data for a grouped bar plot comparing most and least expensive neighborhoods over the years.
   - Creates a grouped bar plot visualizing the data.

3. Analyzing District-wise Rental Price Trends:
   - Groups data by year and district, calculates the mean price, and creates a DataFrame (distr_rents).
   - Extracts prices for different districts from the grouped data.
   - Creates a 3x3 grid of subplots to visualize rental price trends for different districts over the years.
  
4. Predicting Rental Prices for Q3 and Q4 of 2022:
   - Groups data by trimester and year, calculates the mean price, and creates a DataFrame (distr_trim).
   - Prepares data for linear regression modeling.
   - Trains a linear regression model for each quarter and predicts rental prices for Q3 and Q4 of 2022.
   - Creates a 3x3 grid of subplots to visualize rental price trends for each quarter over the years, including predictions for 2022.
  
## Results
  
The main findings of the code can be found in the related Medium post: [Used Cars Price Prediction](https://medium.com/@jaume.bogunaurue/used-cars-price-prediction-d9edb0a4319b).


## Licensing, Authors, Acknowledgements
Credit for the data is attributed to Kaggle. For detailed data licensing and additional descriptive information, you can refer to the Kaggle dataset: [Used Cars Dataset](https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset)https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset).
