import json
import geopandas as gpd
import pandas as pd 
import numpy as np
import pyarrow.parquet as pq
from src.tessellate import tessellate_bounding_box
from src.tile_enrichment import spatial_join
from src.summarization import summarization, compute_time_part
from src.compute_semantic_context import calculate_bow
from src.compute_semantic_locations import merge_locations
from src.clustering import calculate_relevance, assign_taxonomy, clustering_users
from src.most_common_words import save_most_common_words
from src.evaluation import calculate_entropy_and_diversity_per_taxonomy
from src.detect_routine import detect_routine

def main(config):
    # Load the trajectories and tiles GeoDataFrames
    trajectories_gdf = gpd.read_parquet(config['data']['trajectories'])
    trajectories_gdf = trajectories_gdf.sort_values(['uid', 'tid', 'datetime'])
    polygon = gpd.read_parquet(config['data']['tiles'])
    
    # Tessellate the bounding box
    tiles_gdf = tessellate_bounding_box(polygon)
    tiles_gdf.to_parquet('data/tiles.parquet')
    
    # Compute the semantic context
    #tiles_gdf = gpd.read_parquet('data/tiles.parquet')
    
    poi_gdf = gpd.read_parquet(config['data']['poi'])
    landuse_gdf = gpd.read_parquet(config['data']['landuse'])
    pt_gdf = gpd.read_parquet(config['data']['pt'])        

    enriched_tiles_gdf = spatial_join(tiles_gdf, poi_gdf, landuse_gdf, pt_gdf)
    enriched_tiles_gdf.to_parquet('data/enriched_tiles.parquet')    
    
    # Calculate the bag of words for each tile
    #enriched_tiles_gdf = gpd.read_parquet('data/enriched_tiles.parquet')
    tiles_with_context_gdf, feature_names = calculate_bow(enriched_tiles_gdf)
    
    # Merge the locations
    semantic_locations = merge_locations(tiles_with_context_gdf, feature_names)
    semantic_locations.to_parquet('data/semantic_locations.parquet')
    #semantic_locations = gpd.read_parquet('data/semantic_locations.parquet')
    
    # Summarization of trajectories
    joined_gdf = trajectories_gdf.sjoin(semantic_locations)
    summarized_gdf = summarization(joined_gdf,semantic_locations, time_column='datetime', user_id_column='uid', trajectory_id_column='tid')
    summarized_gdf.to_parquet('data/summarized.parquet')
    
    # Calculate the relevance of each tile for each user
    #summarized_gdf = gpd.read_parquet('data/summarized.parquet')
    summarized_gdf = compute_time_part(summarized_gdf, 'uid', 'tid', 'datetime')
    relevance_gdf = calculate_relevance(summarized_gdf, 'uid')    
    #
    # Assign the labels to the clusters
    relevance_gdf['count'] = relevance_gdf.groupby(['uid','new_category'],as_index=False).count()
    relevance_gdf['relevance'] = relevance_gdf.groupby(['uid','new_category'],as_index=False)['relevance'].sum()
    relevance_gdf.to_parquet('data/geolife_beijing_summarized_relevance.parquet')
    
    #relevance_gdf = gpd.read_parquet('data/geolife_beijing_summarized_relevance.parquet')
    relevance_gdf = relevance_gdf.groupby(['uid','new_category'],as_index=False).agg({'relevance':'sum','bag_of_words':'first','context':'first'})
    labeled_gdf = assign_taxonomy(relevance_gdf, 'uid', 'relevance')  
    
    labeled_gdf['bag_of_words'] = semantic_locations['new_bag_of_words']
    labeled_gdf.reset_index(inplace=True)
    labeled_gdf.to_parquet('data/geolife_beijing_summarized_taxonomy.parquet')
    
    # Compute the entropy and diversity for each user
    #labeled_gdf = gpd.read_parquet('data/geolife_beijing_summarized_taxonomy.parquet')
    entropy_diversity_df = calculate_entropy_and_diversity_per_taxonomy(labeled_gdf, user_id_column='uid', taxonomy_column='taxonomy', context_column='new_context', category_column='new_category')
    entropy_diversity_df.to_csv('data/entropy_diversity.csv', index=False)
    
    # Compute routine and non-routine behaviors
    #entropy_diversity_df = pd.read_csv('data/entropy_diversity.csv')
    routine_df, non_routine_df = detect_routine(entropy_diversity_df, taxonomy_column='taxonomy', user_id_column='uid')
    routine_df.to_csv('data/routine.csv', index=False)
    non_routine_df.to_csv('data/non_routine.csv', index=False)
    
    #labeled_gdf = gpd.read_parquet('data/geolife_beijing_summarized_taxonomy.parquet')    
    most_common_df = save_most_common_words(labeled_gdf,feature_names,'uid','taxonomy')
    # Save the DataFrame to a CSV file
    most_common_df.to_csv('data/most_common_words.csv', index=False)
    
    
if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    main(config)
