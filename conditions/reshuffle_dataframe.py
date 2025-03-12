

def reshuffle_fun(df):
    df = df.sample(n=len(df), replace=False)
    # Reset the index of the shuffled DataFrame
    df = df.reset_index(drop=True)
    return df