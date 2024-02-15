# Import necessary libraries
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import geopandas as gpd

def merge_locations(areas, feature_names, threshold=0.8):
    """
    This function calculates the most similar tiles using cosine similarity on the bag of words vectors,
    and assigns the same label and resulting bag of words vector to similar tiles.
    
    Parameters:
    areas (GeoDataFrame): A GeoDataFrame representing the semantically enriched tiles.
    threshold (float): The similarity threshold to consider two tiles as similar.

    Returns:
    GeoDataFrame: A GeoDataFrame with an additional column for the label of similar tiles.
    dict: A dictionary with the indices of similar tiles.
    """
    semantic_locations = areas.copy()

    semantic_locations.reset_index(drop=True, inplace=True)

    bag_of_words_matrix = np.array(semantic_locations['bow'].tolist(), dtype=np.float64)
    similarity_matrix = cosine_similarity(bag_of_words_matrix)
    similar_polygons = {}
        
    for idx, row in semantic_locations.iterrows():
        similar_polys = [i for i, sim in enumerate(similarity_matrix[idx]) if sim > threshold and i != idx]
        if similar_polys:
            similar_polygons[idx] = similar_polys
        
    semantic_locations['new_category'] = -1
    semantic_locations['new_bag_of_words'] = semantic_locations['bow']

    category = 0

    for idx in similar_polygons:
        if semantic_locations.at[idx, 'new_category'] == -1:
            similar_idxs = [idx] + similar_polygons[idx]
            semantic_locations.loc[similar_idxs, 'new_category'] = category
            arrays = semantic_locations.loc[similar_idxs, 'bow'].tolist()
            arrays = np.array([np.array(lst) for lst in arrays])
            new_bag_of_words = np.where((arrays != 0).all(axis=0), arrays.sum(axis=0), 0)
            for sim_idx in similar_polygons[idx]:
                semantic_locations.at[sim_idx, 'new_bag_of_words'] = new_bag_of_words.tolist()
            category += 1
            
    for idx, row in semantic_locations.iterrows():
        if semantic_locations.at[idx, 'new_category'] == -1:
            semantic_locations.at[idx, 'new_category'] = category
            semantic_locations.at[idx, 'new_bag_of_words'] = semantic_locations.at[idx, 'bow']
            category += 1
            
    gdf_merged = semantic_locations.dissolve(by='new_category', as_index=False)
    gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry')
    
    # Compute new context
    gdf_merged['new_context'] = gdf_merged['new_bag_of_words'].apply(lambda x: [feature_names[i] for i in range(len(x)) if x[i] > 0])
    
    return gdf_merged