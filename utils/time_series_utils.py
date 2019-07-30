import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_genre_features_timeseries(df, genres, feature, start_year="1970", end_year="2019", **kwargs):
    """
    Description:
    takes in a dataframe indexed by datetime, a list of genres, and a feature
    Based on these input values, creates an overlayed scatterplot showing how each feature changed over time by genre. 
    """
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    
    for genre in genres:
        current_df = df[df.Super_genre == genre]
        current_df = current_df[start_year:end_year].resample('m').mean()
        
        sns.scatterplot(current_df.index, current_df[feature], data=current_df, alpha=0.66, ax=ax)
    
    plt.legend(genres, fontsize=18)
    plt.title(f"{feature} Over Time by Genre", fontsize=24)
    plt.show()


def convert_to_datetime(t):
    """
    Accepts a string and returns a datetime version if the string is in a valid date format. 
    Otherwise, returns an NaN.
    Meant to be used in an apply call on a pandas series
    """
    try:
        return pd.to_datetime(t)
    except:
        return np.NaN