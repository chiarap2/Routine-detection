# Import necessary libraries
import geopandas as gpd
import pandas as pd

def spatial_join(tiles_gdf, poi_gdf=None, landuse_gdf=None, pt_gdf=None):
    """
    This function performs a spatial join between the tiles GeoDataFrame and three other GeoDataFrames.
    
    Parameters:
    tiles_gdf (GeoDataFrame): A GeoDataFrame representing the tessellated area.
    poi_gdf (GeoDataFrame, optional): A GeoDataFrame representing Points of Interest.
    landuse_gdf (GeoDataFrame, optional): A GeoDataFrame representing land use.
    pt_gdf (GeoDataFrame, optional): A GeoDataFrame representing public transport.

    Returns:
    GeoDataFrame: A GeoDataFrame resulting from the spatial join.
    """
    # Ensure the tiles GeoDataFrame is in the correct format
    assert isinstance(tiles_gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    assert isinstance(poi_gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    assert isinstance(landuse_gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    assert isinstance(pt_gdf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    
    if landuse_gdf is not None:
        landuse_gdf = landuse_gdf[['geometry', 'POI category']]
        landuse_gdf.rename(columns={'POI category':'label'}, inplace=True) 
        
    if poi_gdf is None and landuse_gdf is None and pt_gdf is None:
        return tiles_gdf
    elif poi_gdf is not None and landuse_gdf is None and pt_gdf is None:
        aspects = poi_gdf.copy()
    elif poi_gdf is None and landuse_gdf is not None and pt_gdf is None:
        aspects = landuse_gdf.copy()
    elif poi_gdf is None and landuse_gdf is None and pt_gdf is not None:
        aspects = pt_gdf.copy()
    elif poi_gdf is not None and landuse_gdf is not None and pt_gdf is None:
        aspects = poi_gdf.copy()
        aspects = pd.concat([aspects,landuse_gdf])
    elif poi_gdf is not None and landuse_gdf is None and pt_gdf is not None:
        aspects = poi_gdf.copy()
        aspects = pd.concat([aspects,pt_gdf])
    elif poi_gdf is None and landuse_gdf is not None and pt_gdf is not None:
        aspects = landuse_gdf.copy()
        aspects = pd.concat([aspects,pt_gdf])
    else:
        aspects = poi_gdf.copy()
        aspects = pd.concat([aspects,landuse_gdf])
        aspects = pd.concat([aspects,pt_gdf])
        
    aspects = gpd.GeoDataFrame(aspects, geometry='geometry')
    
    # Perform spatial join with each GeoDataFrame if it is not None
    enriched_tiles = gpd.sjoin(tiles_gdf, aspects)
    enriched_tiles = enriched_tiles[['locationID', 'geometry', 'label']]
    
    return enriched_tiles
