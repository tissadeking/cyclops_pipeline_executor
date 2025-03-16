from sklearn.feature_extraction.text import TfidfVectorizer

def combine_text_data_fun(df, target):
    cols = df.drop(columns=[target]).columns.tolist()
    #df["combined_text"] = df[cols].agg(" ".join, axis=1)
    df["combined_text"] = df[cols].fillna("").agg(" ".join, axis=1)
    # Feature Extraction
    #vectorizer = TfidfVectorizer()
    #X = vectorizer.fit_transform(df["combined_text"])
    return df["combined_text"]