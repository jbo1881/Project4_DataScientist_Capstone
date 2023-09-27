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
- sklearn: which provides tools for machine learning and data analysis. It's used here for linear regression modeling and evaluation.
- datetime
- seaborn
The data was collected form 762,091 used cars scraped from cars.com. The data was ingested on Apr, 2023.

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
Credit for the data is attributed to Kaggle. For detailed data licensing and additional descriptive information, you can refer to the Kaggle dataset: [Used Cars Dataset](https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset).
