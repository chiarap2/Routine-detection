from collections import Counter
import pandas as pd
import numpy as np

def save_most_common_words(relevance_gdf, feature_names, user_id_column='user_id', cluster_column='cluster', terms_column='terms', n_most_common=10):
    
    new_df = pd.DataFrame(columns=['uid'] + [f'{name}_{cluster}' for name in feature_names for cluster in relevance_gdf[cluster_column].unique()])
    
    df = relevance_gdf.groupby([user_id_column, cluster_column], as_index=False)['bag_of_words'].sum()

    # Per ogni utente unico nel DataFrame
    for uid in df[user_id_column].unique():
        # Crea un dizionario per memorizzare i dati di questo utente
        user_data = {'uid': uid}

        # Per ogni cluster
        for taxonomy in df[cluster_column].unique():
            # Se l'utente ha dati per questo cluster
            if ((df[user_id_column] == uid) & (df[cluster_column] == taxonomy)).any():
                # Ottieni i dati per questo utente e cluster
                user_cluster_data = df[(df[user_id_column] == uid) & (df[cluster_column] == taxonomy)].iloc[0]
                
                # Aggiungi le frequenze delle parole ai dati dell'utente
                if type(user_cluster_data['bag_of_words']) != int:
                    for i, name in enumerate(feature_names):
                        user_data[f'{name}_{taxonomy}'] = user_cluster_data['bag_of_words'][i]
        # Crea un DataFrame di una sola riga dal dizionario user_data
        user_df = pd.DataFrame(user_data, index=[0])

        # Aggiungi i dati dell'utente al nuovo DataFrame
        new_df = pd.concat([new_df, user_df], ignore_index=True)

    # Riempie i valori mancanti con zeri
    new_df.fillna(0, inplace=True)

    return new_df
