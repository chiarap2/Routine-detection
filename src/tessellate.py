import geopandas as gpd
from tesspy import Tessellation

def tessellate_bounding_box(geodf,method='square',resolution=18):
    """
    This function takes a GeoDataFrame representing a bounding box and tessellates it using tesspy.
    
    Parameters:
    geodf (GeoDataFrame): A GeoDataFrame representing a bounding box.

    Returns:
    GeoDataFrame: A tessellated GeoDataFrame.
    """
    # Ensure the GeoDataFrame is in the correct format
    assert isinstance(geodf, gpd.GeoDataFrame), "Input must be a GeoDataFrame"
    
    geodf['osm_id'] = 0
    
    # Tessellate the bounding box
    city = Tessellation(geodf)
    
    # Perform the tessellation
    
    if method == 'square':
        tessellation = city.squares(resolution=resolution)
        tessellation.rename(columns={'quadkey':'locationID'},inplace=True)
        
    elif method == 'adaptive_square':
        tessellation = city.adaptive_squares(resolution=resolution)
        tessellation.rename(columns={'quadkey':'locationID'},inplace=True)
        
    elif method == 'voronoi':
        tessellation = city.voronoi()
        tessellation.rename(columns={'voronoi_id':'locationID'},inplace=True)
        
    elif method == 'city_blocks':
        tessellation = city.city_blocks()
        tessellation.rename(columns={'city_block_id':'locationID'},inplace=True)
        
    else:
        raise ValueError("Method not recognized")
    
    return tessellation
