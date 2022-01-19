# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Perameters:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    
     # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on=['id'])
   
    return df


def clean_data(df):
    """Parameters:
    df: Dataframe containing merged content of messages and categories.
       
    Returns:
    df: Dataframe containing cleaned version of input dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing

    category_colnames = row.transform(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
    
        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric, errors='ignore')
        
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Drop rows with a related value of 2
    df = df[df['related'] != 2]
    
    # Drop only rows with all missing values in the category columns
    df.dropna(how='all', subset = category_colnames, inplace=True)
    
    # Remove child alone column since all values are 0 it provides no predicitive value about whether a child is alone or not.
    df.drop('child_alone', axis=1, inplace=True)
    
    #The original column has 15,939 nulls, but since we will not be using this column, we drop it.
    df.drop('original', axis=1, inplace=True)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df
    
    
def save_data(df, database_filename):
    """Parameters:
    df: Dataframe containing cleaned version of merged data.
    database_filename: string. Filename for the database.
     
    Returns: None
    """ 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()