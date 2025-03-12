from sklearn.impute import SimpleImputer

def fill_empty_fun(df):
    # Imputer for numerical columns (replace NaN with mean)
    num_imputer = SimpleImputer(strategy='mean')

    # Imputer for categorical columns (replace NaN with most frequent value)
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Apply imputers
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

