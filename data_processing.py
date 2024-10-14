import pandas as pd
import streamlit as st
import os


@st.cache_data
def process_datasets() -> pd.DataFrame:
    # Extract datasets
    df = None

    for name in ['simon', 'marcos', 'luz', 'monica_pablo']:
        new = pd.read_csv(os.path.join('datasets', f'NetflixViewingHistory_{name}.csv'))
        new['Name'] = name
        df = pd.concat([df, new], axis=0)

    # Replace names
    df['Name'].replace(
        {
            'simon': 'Simón',
            'marcos': 'Marcos',
            'luz': 'Luz',
            'monica_pablo': 'Mónica y Pablo'
        }, inplace=True
    )

    # Extract Title, Season & Episode
    def extract_attrs(row):
        # Extract title
        title = row['Title']
        splits = title.split(': ')
        
        if len(splits) == 0:
            return [None, None, None]
        if len(splits) == 1:
            return [title, None, None]
        if len(splits) == 2:
            return [splits[0], None, splits[1]]
        if len(splits) == 3:
            return splits
        
        return splits[0], splits[1], ': '.join(splits[2:])
        

    df['attrs'] = df.apply(extract_attrs, axis=1)

    df['Title'] = df['attrs'].apply(lambda x: x[0])
    df['Season'] = df['attrs'].apply(lambda x: x[1])
    df['Episode'] = df['attrs'].apply(lambda x: x[2])

    df.drop(columns=['attrs'], inplace=True)

    # Add IsSeries column
    df['IsSeries'] = df.apply(lambda row: True if row['Episode'] is not None else False, axis=1)

    # Parse Date to datetime column
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort columns
    df = df[['Name', 'Title', 'Season', 'Episode', 'IsSeries', 'Date']]

    return df


# conda deactivate
# source .infovis_venv/bin/activate
# .infovis_venv/bin/python data_processing.py
if __name__ == "__main__":
    df = process_datasets()

    print(df)