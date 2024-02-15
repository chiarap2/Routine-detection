import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
import ast

def calculate_relevance(time_spent_gdf, user_id_column='user_id'):
    """
    This function calculates the relevance of each tile for each user.
    """
    # Calculate the total time spent by each user
    time_spent_gdf['time_spent'] = time_spent_gdf['time_spent'].astype('timedelta64[s]').astype(int)
    time_spent_gdf['time_spent'] = time_spent_gdf['time_spent'].fillna(0)
        
    # Calculate the relevance
    time_spent_gdf['relevance'] = time_spent_gdf['time_spent'] / time_spent_gdf.groupby(user_id_column)['time_spent'].transform('sum')
    
    return time_spent_gdf

def cluster_tiles(relevance_gdf, user_id_column='user_id', n_clusters=3):
    """
    This function clusters the tiles for each user based on the bag of words vectors using KMeans.
    """
    # Initialize a KMeans object
    kmeans = KMeans(n_clusters=n_clusters)
    
    # Cluster the tiles for each user
    relevance_gdf['cluster'] = relevance_gdf.groupby(user_id_column)['bag_of_words'].transform(lambda x: kmeans.fit_predict(np.array(x.tolist())))
    
    return relevance_gdf

def assign_labels_to_clusters(group):
    """
    This function assigns the labels "MVP", "FVP" and "EVP" to the first three clusters for each user, sorted by relevance.
    """
    # Define the labels
    labels = ["MVP", "FVP", "EVP"]
    
    # Assign the labels
    group['label'] = labels[:len(group)]
    
    return group

def assign_labels(clusters_gdf, user_id_column='user_id', relevance_column='relevance', cluster_id_column='cluster'):
    """
    This function assigns the labels "MVP", "FVP" and "EVP" to the first three clusters for each user, sorted by relevance.
    
    Parameters:
    clusters_gdf (GeoDataFrame): A GeoDataFrame representing the clusters.
    user_id_column (str, optional): The column in the clusters GeoDataFrame that contains the user ids.
    relevance_column (str, optional): The column in the clusters GeoDataFrame that contains the relevance.

    Returns:
    GeoDataFrame: A GeoDataFrame with an additional column for the labels.
    """
    
    # Sort the clusters by relevance
    
    clusters_gdf = clusters_gdf.groupby([user_id_column, cluster_id_column]).sum()
    
    clusters_gdf.sort_values([user_id_column, relevance_column], ascending=[True, False], inplace=True)
    
    # Assign the labels
    clusters_gdf['label'] = clusters_gdf.groupby(user_id_column, as_index=False).apply(assign_labels_to_clusters)
    
    return clusters_gdf

def cluster_tiles_per_user(relevance_gdf, user_id_column='user_id', bow_column='bag_of_words'):
    """
    This function clusters the tiles for each user based on the bag of words vectors using KMeans,
    choosing the number of clusters based on the silhouette and elbow methods.
    
    Parameters:
    relevance_gdf (GeoDataFrame): A GeoDataFrame representing the relevance of the tiles.
    user_id_column (str, optional): The column in the relevance GeoDataFrame that contains the user ids.
    bow_column (str, optional): The column in the relevance GeoDataFrame that contains the bag of words vectors.

    Returns:
    GeoDataFrame: A GeoDataFrame with an additional column for the cluster labels.
    """
    # Initialize the range of possible number of clusters
    range_n_clusters = list(range(2, 7))
    
    for user_id, group in relevance_gdf.groupby(user_id_column):
        X = np.array(group[bow_column].tolist())
        
        # Calculate the silhouette scores and SSE for each number of clusters
        silhouette_scores = []
        sse = []
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            sse.append(clusterer.inertia_)
        
        # Calculate the maximum silhouette score and the elbow point
        max_silhouette_score = max(silhouette_scores)
        elbow_point = np.argmin(np.gradient(np.gradient(sse)))
        
        # Choose the number of clusters
        if max_silhouette_score == elbow_point:
            n_clusters = np.argmax(silhouette_scores) + 2
        else:
            n_clusters = np.argmax(silhouette_scores) + 2
        
        # Cluster the tiles for the current user
        clusterer = KMeans(n_clusters=n_clusters)
        relevance_gdf.loc[group.index, 'cluster'] = clusterer.fit_predict(X)
    
    return relevance_gdf

def assign_taxonomy(relevance_gdf, user_id_column='user_id',relevance_column='relevance'):
    """
    This function assigns a taxonomy to the clusters based on the relevance within each cluster.
    
    Parameters:
    relevance_gdf (GeoDataFrame): A GeoDataFrame representing the relevance of the tiles.
    user_id_column (str, optional): The column in the relevance GeoDataFrame that contains the user ids.
    cluster_column (str, optional): The column in the relevance GeoDataFrame that contains the cluster labels.
    relevance_column (str, optional): The column in the relevance GeoDataFrame that contains the relevance.

    Returns:
    GeoDataFrame: A GeoDataFrame with an additional column for the taxonomy.
    """
    # Define the labels
    labels = ["Insignificant locations", "Sporadic locations", "Transit locations", "Significant locations"]
     
    # Calculate the percentiles
    percentiles = np.percentile(relevance_gdf[relevance_column], [25, 50, 75])

    # Assign the taxonomy to the clusters
    relevance_gdf['taxonomy'] = pd.cut(
        relevance_gdf[relevance_column], 
        bins=[-np.inf] + list(percentiles) + [np.inf], 
        labels=labels
    )
    
    return relevance_gdf

def clustering_users(most_common_df,user_id_column='uid',taxonomy_column='taxonomy'):
    """
    This function clusters the users based on the context and the taxonomy of the clusters.
    
    Parameters:
    merged_df (DataFrame): A DataFrame representing the merged GeoLife and enriched tiles data.
    user_id_column (str, optional): The column in the merged DataFrame that contains the user ids.
    context_column (str, optional): The column in the merged DataFrame that contains the context.
    taxonomy_column (str, optional): The column in the merged DataFrame that contains the taxonomy.

    Returns:
    DataFrame: A DataFrame with an additional column for the user clusters.
    """
    
    X = most_common_df.iloc[:,1:].values
    
    # Cluster the users based on the bag of words using k-means
    
    # Initialize the range of possible number of clusters
    range_n_clusters = list(range(2, 7))
    
    # Calculate the silhouette scores and SSE for each number of clusters
    silhouette_scores = []
    sse = []
    
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        # normalize the data
           
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        sse.append(clusterer.inertia_)
        
    # Calculate the maximum silhouette score and the elbow point
    max_silhouette_score = max(silhouette_scores)
    elbow_point = np.argmin(np.gradient(np.gradient(sse)))
    
    # Choose the number of clusters
    if max_silhouette_score == elbow_point:
        n_clusters = np.argmax(silhouette_scores) + 2
    else:
        n_clusters = np.argmax(silhouette_scores) + 2
        
    print(n_clusters)
    # Cluster the users
    clusterer = KMeans(n_clusters=n_clusters)
    most_common_df['user_cluster'] = clusterer.fit_predict(X)
    
    return most_common_df