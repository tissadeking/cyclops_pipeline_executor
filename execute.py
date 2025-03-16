from conditions.save_model import save_model_fun
from conditions.combine_dataframe import combine_df_fun
from conditions.combine_generated_data import combine_gen_data_fun
from conditions.drop_empty_rows import drop_empty_fun
from conditions.fill_empty_rows import fill_empty_fun
from conditions.encoding import one_hot_encode, label_encode, encode_target
from conditions.reshuffle_dataframe import reshuffle_fun
from conditions.scaling import scaling_fun, scaling_fun_selected_cols
from conditions.train_test_split import split_fun
from conditions.get_metrics import get_metrics_reg_fun, get_metrics_class_fun, get_metrics_forecast_fun, get_metrics_var_forecast_fun
from conditions.combine_text_data import combine_text_data_fun
from conditions.convert_np_array import convert_np_array_fun
from conditions.extract_x_y import extract_x_y_fun, extract_x_y_text_fun
from anomaly_detection.autoencoders import autoencode_fun
from anomaly_detection.isolation_forest import isolation_fun
from anomaly_detection.one_class_svm import one_class_svm_fun
from classification.num_classify import pac_model_fun, log_model_fun, mnb_model_fun, mlpc_model_fun, svc_model_fun, dt_model_fun, kn_model_fun, rf_model_fun
from classification.text_classify import pac_model_text, log_model_text, mnb_model_text, mlpc_model_text, svc_model_text, dt_model_text, kn_model_text, rf_model_text
from classification.text_classify_cross_val import pac_model_cross_val, log_model_cross_val, mnb_model_cross_val, mlpc_model_cross_val, svc_model_cross_val, dt_model_cross_val, kn_model_cross_val, rf_model_cross_val
from clustering.cluster_models import affinity_fun, dbscan_fun, gmm_fun,heirarchy_fun, km_fun, optics_fun
from regression.regressor import linear_model_fun, lasso_model_fun, ridge_model_fun, svr_model_fun, dt_reg_model_fun, nn_model_fun, rf_reg_model_fun
from time_series_forecasting.forecast import ar_fun, arima_fun, var_fun
from inference.inference import inference_fun, inference_forecast_ar, inference_forecast_arima, inference_var_forecast, inference_km, inference_dbscan


'''pipeline = {
            "intent_type": "analytical",
            "data": ["dataframe 1", "dataframe 2", "...."],
            "task": "regression, etc",
            "task_description": {
                "training": "yes/no",
                "inference": {
                    "infer": "yes/no",
                    "model_name": "filename"
                },
                "save_model": "yes/no"
            },
            "model": "",
            "target": "target variable",
            "specific_columns": ["feature 1", "feature 2", "feature 3"],
            "test_fraction": 0.2,
            "pre-conditions": [],
            "scaling_bounds": [-1, 1],
            "post-conditions": [],
            "pipeline_id": "XYTHSI33"
        }'''

def execute_fun(pipeline):
    global model, split_data, metric_val, y_pred, conv_extr, X

    df = pipeline['data'][0]
    target = pipeline['target']
    pre_conditions = pipeline['pre-conditions']
    post_conditions = pipeline['post-conditions']
    specific_columns = pipeline['specific_columns']
    task = pipeline['task']
    pipeline_model = pipeline['model']


    #PRE-CONDITIONS
    if 'combine_text_data' in pre_conditions:
        X = combine_text_data_fun(df, target)
    if 'encode_target' in pre_conditions:
        df[target] = encode_target(df[target])
    if 'convert_np_array' in pre_conditions:
        conv_extr = convert_np_array_fun(df, target)
    elif 'extract_x_y_text' in pre_conditions:
        conv_extr = extract_x_y_text_fun(df, target, X)
    elif 'extract_x_y' in pre_conditions:
        conv_extr = extract_x_y_fun(df, target)

    if 'train_split' in pre_conditions:
        split_data = split_fun(conv_extr[0], conv_extr[1], pipeline['test_fraction'])

    if 'combine_dataframe' in pre_conditions:
        df = combine_df_fun(pipeline['data'])
    elif 'combine_generated_data' in pre_conditions:
        df = combine_gen_data_fun(pipeline['data'])

    if 'drop_empty_rows' in pre_conditions:
        df = drop_empty_fun(df)
    elif 'fill_empty_rows' in pre_conditions:
        df = fill_empty_fun(df)

    if 'one_hot_encoding' in pre_conditions:
        df = one_hot_encode(df, specific_columns)
    elif 'label_encoding' in pre_conditions:
        df = label_encode(df, specific_columns)

    if 'reshuffle_dataframe' in pre_conditions:
        df = reshuffle_fun(df)

    if 'scaling' in pre_conditions:
        df = scaling_fun(df, pipeline['scaling_bounds'])
    elif 'scaling_selected_cols' in pre_conditions:
        df = scaling_fun_selected_cols(df, pipeline['scaling_bounds'],
                                       specific_columns)



    #TASKS and MODELS
    #TRAINING
    if 'anomaly_detection' in task:
        if pipeline_model == 'autoencoders':
            autoencode_fun(df, pipeline['data'][1])
        elif pipeline_model == 'isolation_forest':
            isolation_fun(df)
        elif pipeline_model == 'one_class_svm':
            one_class_svm_fun(df)

    elif 'num_classification' in task:
        if pipeline_model == 'pac':
            model = pac_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'log':
            model = log_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'mnb':
            model = mnb_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'mlpc':
            model = mlpc_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'svc':
            model = svc_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'dt':
            model = dt_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'kn':
            model = kn_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'rf':
            model = rf_model_fun(split_data[0], split_data[2])

    elif 'text_classification' in task:
        if pipeline_model == 'pac':
            model = pac_model_text(split_data[0], split_data[2])
        elif pipeline_model == 'log':
            model = log_model_text(split_data[0], split_data[2])
        elif pipeline_model == 'mnb':
            model = mnb_model_text(split_data[0], split_data[2])
        elif pipeline_model == 'mlpc':
            model = mlpc_model_text(split_data[0], split_data[2])
        elif pipeline_model == 'svc':
            model = svc_model_text(split_data[0], split_data[2])
        elif pipeline_model == 'dt':
            model = dt_model_text(split_data[0], split_data[2])
        elif pipeline_model == 'kn':
            model = kn_model_text(split_data[0], split_data[2])
        elif pipeline_model == 'rf':
            model = rf_model_text(split_data[0], split_data[2])

    elif 'text_classification_cross_val' in task:
        if pipeline_model == 'pac':
            model = pac_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline_model == 'log':
            model = log_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline_model == 'mnb':
            model = mnb_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline_model == 'mlpc':
            model = mlpc_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline_model == 'svc':
            model = svc_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline_model == 'dt':
            model = dt_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline_model == 'kn':
            model = kn_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline_model == 'rf':
            model = rf_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])

    elif 'clustering' in task:
        X = df[pipeline['specific_columns'][0]].tolist()
        if pipeline_model == 'affinity':
            model = affinity_fun(X)
        elif pipeline_model == 'dbscan':
            model = dbscan_fun(X)
        elif pipeline_model == 'gmm':
            model = gmm_fun(X)
        elif pipeline_model == 'km':
            model = km_fun(X)
        elif pipeline_model == 'heirarchy':
            model = heirarchy_fun(X)
        elif pipeline_model == 'optics':
            model = optics_fun(X)

    elif 'regression' in task:
        if pipeline_model == 'linear':
            model = linear_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'lasso':
            model = lasso_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'ridge':
            model = ridge_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'nn':
            model = nn_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'svr':
            model = svr_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'dt':
            model = dt_reg_model_fun(split_data[0], split_data[2])
        elif pipeline_model == 'rf':
            model = rf_reg_model_fun(split_data[0], split_data[2])

    elif 'time_series_forecasting' in task:
        X = df[pipeline['specific_columns'][0]].tolist()
        if pipeline_model == 'ar':
            model = ar_fun(X)
        elif pipeline_model == 'arima':
            model = arima_fun(X)
        elif pipeline_model == 'var':
            model = var_fun(df)

    #INFERENCE
    to_infer = pipeline['task_description']['inference']['infer']
    model_name = pipeline['task_description']['inference']['model_name']
    if 'num_classification' in task or \
        'text_classification_cross_val' in task or \
        'text_classification' in task:
        if 'yes' in to_infer:
            y_pred = inference_fun(split_data[1], model_name)

    if 'clustering' in task and (pipeline_model == 'dbscan' or
        pipeline_model == 'affinity' or pipeline_model == 'optics'):
        if 'yes' in to_infer:
            y_pred = inference_dbscan(split_data[1], model_name)
    elif 'clustering' in task:
        if 'yes' in to_infer:
            y_pred = inference_km(split_data[1], model_name)

    if 'time_series_forecasting' in task:
        if 'yes' in to_infer:
            if pipeline_model == 'ar':
                y_pred = inference_forecast_ar(split_data[1], model_name)
            elif pipeline_model == 'arima':
                y_pred = inference_forecast_arima(split_data[1], model_name)
            elif pipeline_model == 'var':
                y_pred = inference_var_forecast(split_data[1], model_name)


    #POST-CONDITIONS
    if 'save_model' in post_conditions:
        save_model_fun(model, pipeline_model)

    if 'get_metrics_class' in post_conditions:
        metric_val = get_metrics_class_fun(split_data[1], split_data[3],
                                           model)
    elif 'get_metrics_reg' in post_conditions:
        metric_val = get_metrics_reg_fun(split_data[1], split_data[3],
                                           model)
    elif 'get_metrics_forecast_fun' in post_conditions:
        metric_val = get_metrics_forecast_fun(split_data[1], split_data[3],
                                           model)
    elif 'get_metrics_var_forecast_fun' in post_conditions:
        metric_val = get_metrics_var_forecast_fun(df, df,
                                         model)




















