
import pandas as pd
from sklearn.model_selection import train_test_split

def data_cleaner():
    file_path = "./bbc-news-data - bbc-news-data.csv"
    df = pd.read_csv(file_path)
    # print(df.head())
    
    df_clean = df.iloc[:, 0].str.split("\t", expand=True)
    df_clean.columns = ["category", "filename", "title", "content"]
    df_clean = df_clean[["category", "content"]]

    # print(df_clean.head())
    return df_clean

def data_separator(df_clean):
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

    train_file_path = "./data/train_news.csv"
    test_file_path = "./data/test_news.csv"

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    train_file_path, test_file_path


data_separator(data_cleaner())