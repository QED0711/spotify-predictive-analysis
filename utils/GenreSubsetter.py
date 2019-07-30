import pandas as pd

class GenreSubsetter:
    
    @staticmethod
    def get_genre_subset(df, genre_list):
        """
        Subsets a dataframe on multiple Super_genres
        """
        subset = df[df.Super_genre == genre_list[0]]
        for i in range(1, len(genre_list)):
            subset = pd.concat([subset, df[df.Super_genre == genre_list[i]]])

        return subset