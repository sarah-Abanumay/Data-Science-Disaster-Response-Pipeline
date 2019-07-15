import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
        messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df
    pass


def clean_data(df):
        # split the categories columns into multiple columns
    categories = df['categories'].str.split(';', expand=True)

    # rename columns
    row = categories.iloc[1]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # replace original values into 1 and 0
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))

    # replace the old categories column
    df.drop('categories', axis = 1, inplace = True)
    df = df.join(categories)
    # drop duplicates
    df = df.drop_duplicates()
    return df
    pass


def save_data(df, database_filename):
    engine = create_engine('sqlite:///../workspace/data/DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False,if_exists='replace')
    pass  


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