from sklearn.preprocessing import MinMaxScaler

#scale columns in dataframe
def scaling_fun(df, bounds):
    # Select only numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler(feature_range=(bounds[0], bounds[1]))
    # Scale only numerical columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

#select specific cols to scale
def scaling_fun_selected_cols(df, bounds, cols):
    scaler = MinMaxScaler(feature_range=(bounds[0], bounds[1]))
    # Scale only numerical columns
    df[cols] = scaler.fit_transform(df[cols])
    return df

