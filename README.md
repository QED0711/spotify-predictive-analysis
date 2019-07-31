# Spotify's Data Problem

## Contents

- [Introduction](#Introduction)
    - [Problem statement](#Problem-statement)
    - [Dataset](#Dataset)
- [Analysis](#Analysis)
    - [Data Cleaning](#Data-Cleaning)
    - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
    - [Time Series Analysis](#Time-Series-Analysis)
    - [Linear Regression](#Linear-Regression)
    - [Logistic Regression](#Logistic-Regression)
- [Responsibilties](#Responsibilities)
- [Summary of Files](#Files-summary)

## Introduction
<img src="https://dl2.macupdate.com/images/icons256/33033.png?d=1562168849" alt="spotify logo" style="margin:0 auto; width=300px;" >
### Problem Statement
Is the data that spotify put to describe and classify songs feasible for predictive or classifying analysis?


### Dataset
The data used for this project was taken from kaggle in conjunction with our own Spotify API query to complete the necessary data needed for our intended analysis.
 
The features from the dataset are described in detail at the [Spotify documentation](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)
 
## Analysis

### Data Cleaning
The data from kaggle was very clean. As mentioned prior, we just needed supplemental information hence the API query. Our project needed the 'time' measure/feature to conduct time series analysis.

The final dataset has no text features that would need to be dropped nor further analysis for sentiment.

### Exploratory Data Analysis

Fot the initial linear regression analysis, correlation was explored and the result showed very few high correlations. Based on the correlations, the target chosen was 'Energy' as it has the most and highest correlations on other features.


### Time Series Analysis
Our time series analysis is broken down into two parts. First, we look at how the number of tracks/albums rleased over time changes, and how certain features in our dataset change in that same time period. Second, we incorperate information from our genre analysis (logistic regression), and see how certain features change over time with respect to specific genres. 

### Linear Regression
The linear regression analysis was done on all the features to include dummies for all categorical features. The initial result was acceptable at an adjusted R-squared value of 0.77. From this result, the target was transformed to make its distribution closer to normal. Some features were also transformed for similar purpose.

After the transformation, the model improved the scores by 0.03 for all inclusive, training, and test sets of data.

Lastly, Ridge Regression was applied to the data with the optimal alpha of 0.001 and a resulting Adjusted R-Squared of 0.801

### Logistic Regression
Using logistic regression, we attempt to predict genres based on the provided features. We first attempt to make predictions between pairs of genres and record the varying results. Following this, we attempt a multinomial logistic regression to predict between multiple genres.     

## Responsibilities

The data selection/retrieval and Exploratory Data Analysis was a joint effort; the Spotify Query was completed by Quinn which completed our dataset.

Time Series Analysis and Logistic Regression was completed by Quinn.

Linear Regression was completed by Allan.

The project presentation was a joint effort.

## Summary of Files

All .png files were used for project presentation.