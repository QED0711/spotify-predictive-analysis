import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil


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
        
        sns.scatterplot(current_df.index, current_df[feature], data=current_df, alpha=0.4, ax=ax)
    
    plt.legend(genres, fontsize=18)
    plt.title(f"{feature} Over Time by Genre", fontsize=24)
    plt.xlabel("Release Date", fontsize=18)
    plt.ylabel(feature, fontsize=18)
    plt.show()

def ts_plot_scatter_group(df, features, genres, start="1970", stop="2018"):
    fig = plt.figure(figsize=(16,8))

    for i in range(len(features)):
        ax = fig.add_subplot(ceil(len(features) / 2), 2, i + 1)

        feature = features[i]

        for genre in genres:
            current_df = df[df.Super_genre == genre]
            current_df = current_df[start:stop].resample('m').mean()
            
            sns.scatterplot(current_df.index, current_df[feature], data=current_df, alpha=0.4, ax=ax)

        plt.legend(genres)
        plt.xlabel("Release Date")
        plt.title(f"{feature}: {start}-{stop}", fontsize=18)

    plt.tight_layout()
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



def plot_feature(ts_df, feature, start="1950", stop="2018"):
    plt.figure(figsize=(16,6))
    ts_df[start:stop][feature].plot(alpha=0.5)
    ts_df[start:stop].resample("y").mean()[feature].plot()
    
    plt.title(f"{feature} from {start}-{stop}", fontsize=24)
    plt.xlabel("Release Date", fontsize=18)
    plt.ylabel(feature, fontsize=18)
    
    plt.legend(["Months", "Years"])
    
    plt.show()


def ts_plot_group(df, features, start="1950", stop="2018"):
    
    fig = plt.figure(figsize=(16, 8))
    
    for i in range(len(features)):
        
        fig.add_subplot(ceil(len(features) / 2), 2, i + 1)

        feature = features[i]
        df[start:stop][feature].plot(alpha=0.5)
        df[start:stop].resample("y").mean()[feature].plot()

        plt.title(f"{feature}: {start}-{stop}", fontsize=18)
        plt.xlabel("Release Date", fontsize=12)
        plt.ylabel(f"{feature} (average)", fontsize=12)

    plt.tight_layout()
    plt.show()
