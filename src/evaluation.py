# %%
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import numpy as np
from scipy.stats import entropy

def calculate_entropy_and_diversity_per_taxonomy(df, user_id_column='user_id', taxonomy_column='taxonomy', context_column='new_context', category_column='new_category'):
    """
    This function calculates the average entropy and diversity of contexts for each user.
    
    Parameters:
    df (DataFrame): A DataFrame containing the user ids and contexts.
    user_id_column (str, optional): The column in the DataFrame that contains the user ids.
    context_column (str, optional): The column in the DataFrame that contains the contexts.

    Returns:
    DataFrame: A DataFrame with the average entropy and diversity for each user.
    """
    # Calculate the entropy for each user
    entropy_df = df.groupby([user_id_column,taxonomy_column])[category_column].apply(lambda x: entropy(x.value_counts().values))
    
    # Calculate the diversity for each user
    diversity_df = df.groupby([user_id_column,taxonomy_column])[context_column].apply(lambda x: [i for sublist in x for i in sublist])
    diversity_df = diversity_df.apply(lambda x: len(set(x)) if type(x)==list else 0)    
    # Compute the percentage of diversity
    diversity_df = diversity_df / diversity_df.groupby(user_id_column).transform('sum')
    
    # Combine the entropy and diversity into a single DataFrame
    result = pd.DataFrame({'entropy': entropy_df, 'diversity': diversity_df})

    return result