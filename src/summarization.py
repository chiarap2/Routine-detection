import geopandas as gpd
import pandas as pd
import numpy as np

def summarization(joined_gdf,tiles_gdf, time_column='time', user_id_column='user_id', trajectory_id_column='trajectory_id'):
    """
    This function associates each point of the trajectories in a GeoDataFrame with the tiles obtained with merge_locations,
    calculates the time spent in each tile, and groups by user id and trajectory id.
    
    Parameters:
    trajectories_gdf (GeoDataFrame): A GeoDataFrame representing the trajectories.
    tiles_gdf (GeoDataFrame): A GeoDataFrame representing the tiles.
    time_column (str, optional): The column in the trajectories GeoDataFrame that contains the time.
    user_id_column (str, optional): The column in the trajectories GeoDataFrame that contains the user ids.
    trajectory_id_column (str, optional): The column in the trajectories GeoDataFrame that contains the trajectory ids.

    Returns:
    GeoDataFrame: A GeoDataFrame with an additional column for the time spent in each tile.
    """
    # Ensure the trajectories GeoDataFrame is in the correct format
    #assert isinstance(trajectories_gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    #assert isinstance(tiles_gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    
    # Perform a spatial join between the trajectories and the tiles
    #joined_gdf = trajectories_gdf.sjoin(tiles_gdf,how='left')
    
    # Sort the GeoDataFrame by user id, trajectory id and time
    joined_gdf = joined_gdf.sort_values([user_id_column, trajectory_id_column, time_column])
    
    # Calculate the time difference in seconds between two consecutive points
    joined_gdf['time_diff'] = joined_gdf.groupby([user_id_column, trajectory_id_column])[time_column].diff()
    
    # Create a 'day_of_week' column
    joined_gdf['day_of_week'] = joined_gdf[time_column].dt.dayofweek
    
    # Create an 'hour' column
    joined_gdf['hour'] = joined_gdf[time_column].dt.hour

    # Group by user id, trajectory id and tile category
    grouped = joined_gdf.groupby([user_id_column, trajectory_id_column, 'new_category'])
    
    # Calculate the total time spent in each tile
    time_spent = grouped['time_diff'].sum()
    
    # Get the list of days and hours for each group
    days = grouped['day_of_week'].apply(list)
    hours = grouped['hour'].apply(list)
    
    # Create a new DataFrame with the calculated values
    result = pd.DataFrame({
        'time_spent': time_spent,
        'days': days,
        'hours': hours,
        'bag_of_words': grouped['new_bag_of_words'].first(),
        'context': grouped['new_context'].first()
    })    
     
    result = result.reset_index()
    
    result = result.dropna(subset=['new_category'])
    
    result = result.merge(tiles_gdf[['new_category','geometry']], on='new_category')
    result = gpd.GeoDataFrame(result, geometry='geometry')
    
    return result

def compute_time_part(summarized_gdf, user_id_column='uid',trajectory_id_column='tid',time_column='datetime'):
    
    summarized_gdf = summarized_gdf.sort_values([user_id_column, trajectory_id_column, time_column])
    summarized_gdf['time_diff'] = summarized_gdf.groupby([user_id_column, trajectory_id_column])[time_column].diff()
    
    # Create a 'day_of_week' column
    summarized_gdf['day_of_week'] = summarized_gdf[time_column].dt.day_name()
    
    # Create an 'hour' column
    summarized_gdf['hour'] = summarized_gdf[time_column].dt.hour

    # Group by user id, trajectory id and tile category
    grouped = summarized_gdf.groupby([user_id_column, trajectory_id_column, 'new_category'])
    
    # Calculate the total time spent in each tile
    time_spent = grouped['time_diff'].sum()
    
    # Get the list of days and hours for each group
    days = grouped['day_of_week'].apply(list)
    hours = grouped['hour'].apply(list)
    
    # Create a new DataFrame with the calculated values
    result = pd.DataFrame({
        'time_spent': time_spent,
        'days': days,
        'hours': hours,
        'bag_of_words': grouped['new_bag_of_words'].first(),
        'context': grouped['new_context'].first()
    })    
     
    result = result.reset_index()
    
    return result