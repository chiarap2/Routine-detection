# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import geopandas as gpd

def calculate_bow(tiles_gdf, category_column='label'):
    """
    This function assigns to each semantically enriched tile a list of all the categories that fall into the tile 
    and calculates the corresponding bag of words vector.
    
    Parameters:
    tiles_gdf (GeoDataFrame): A GeoDataFrame representing the semantically enriched tiles.
    category_column (str, optional): The column in the GeoDataFrame that contains the categories.

    Returns:
    GeoDataFrame: A GeoDataFrame with an additional column for the bag of words vector.
    """
    # Ensure the tiles GeoDataFrame is in the correct format
    assert isinstance(tiles_gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    
    # Drop nan values
    tiles_gdf = tiles_gdf.dropna()
    
    # Preprocessing category column
    tiles_gdf[category_column] = tiles_gdf[category_column].str.replace(' ', '_') 
    tiles_gdf[category_column] = tiles_gdf[category_column].str.replace(',', '_')
    tiles_gdf[category_column] = tiles_gdf[category_column].str.replace('-', '_')
    
    tiles_gdf.set_index('locationID', inplace=True)
    
    # Create a new column 'context' that contains a list of all categories for each tile
    tiles_gdf['context'] = tiles_gdf.groupby('locationID')[category_column].apply(lambda x: ' '.join(x))
    
    # Initialize a CountVectorizer
    vectorizer = CountVectorizer()
    
    # Compute the bag of words for each tile
    X = vectorizer.fit_transform(tiles_gdf['context'])
    bag_of_words = X.toarray()
    tiles_gdf['bow'] = bag_of_words.tolist()
    
    tiles_gdf = tiles_gdf.reset_index()
    tiles_gdf = tiles_gdf.drop_duplicates(subset='locationID')
    tiles_gdf = tiles_gdf[['locationID', 'geometry', 'context', 'bow']]
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return tiles_gdf, feature_names
