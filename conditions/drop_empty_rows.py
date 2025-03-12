

def drop_empty_fun(df):
    columns = df.columns.tolist()
    for col in columns:
        df.dropna(subset=[col], inplace=True)
    return df