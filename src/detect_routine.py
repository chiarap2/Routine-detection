import pandas as pd

def detect_routine(df, taxonomy_column='taxonomy', user_id_column='uid'):
    """
    This function detects the routine of each user based on the taxonomy.
    
    Parameters:
    df (DataFrame): A DataFrame containing the user ids and taxonomies.
    taxonomy_column (str, optional): The column in the DataFrame that contains the taxonomies.
    user_id_column (str, optional): The column in the DataFrame that contains the user ids.
    
    Returns:
    DataFrame: A DataFrame with the routine of each user.
    """
    
    df.reset_index(inplace=True)
    df.sort_values(by=[user_id_column,'entropy'],inplace=True) 
    
    min_entropy = df.groupby(user_id_column).first()
    max_entropy = df.groupby(user_id_column).last()
    
    max_entropy_sig = max_entropy[max_entropy['taxonomy']=='Significant locations']
    max_entropy_trans = max_entropy[max_entropy['taxonomy']=='Transit locations']
    min_entropy_sig = min_entropy[min_entropy['taxonomy']=='Significant locations']
    min_entropy_trans = min_entropy[min_entropy['taxonomy']=='Transit locations']

    # Create a DataFrame with the routine of each user concating min_entropy_sig and min_entropy_trans      
    routine_df = pd.concat([min_entropy_sig,min_entropy_trans])
    routine_df = routine_df[['uid','taxonomy','entropy','diversity']]
    
    # Create a DataFrame with the non routine of each user concating max_entropy_sig and max_entropy_trans
    non_routine_df = pd.concat([max_entropy_sig,max_entropy_trans]) 
    non_routine_df = non_routine_df[['uid','taxonomy','entropy','diversity']]
    
    return routine_df, non_routine_df
                
                        
    
