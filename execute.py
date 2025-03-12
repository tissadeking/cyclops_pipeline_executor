from conditions.save_model import save_model_fun
from conditions.combine_dataframe import combine_df_fun
from conditions.combine_generated_data import combine_gen_data_fun
from conditions.drop_empty_rows import drop_empty_fun
from conditions.fill_empty_rows import fill_empty_fun
from conditions.encoding import one_hot_encode, label_encode
from conditions.reshuffle_dataframe import reshuffle_fun
from conditions.scaling import scaling_fun, scaling_fun_selected_cols
from conditions.train_test_split import split_fun
from conditions.get_metrics import get_metrics_reg_fun, get_metrics_class_fun, get_metrics_forecast_fun, get_metrics_var_forecast_fun
from anomaly_detection.autoencoders import autoencode_fun
from anomaly_detection.isolation_forest import isolation_fun
from anomaly_detection.one_class_svm import one_class_svm_fun
from classification.num_classify import pac_model_fun, log_model_fun, mnb_model_fun, mlpc_model_fun, svc_model_fun, dt_model_fun, kn_model_fun, rf_model_fun
from classification.text_classify import pac_model_text, log_model_text, mnb_model_text, mlpc_model_text, svc_model_text, dt_model_text, kn_model_text, rf_model_text
from classification.text_classify_cross_val import pac_model_cross_val, log_model_cross_val, mnb_model_cross_val, mlpc_model_cross_val, svc_model_cross_val, dt_model_cross_val, kn_model_cross_val, rf_model_cross_val
from clustering.cluster_models import affinity_fun, dbscan_fun, gmm_fun,heirarchy_fun, km_fun, optics_fun
from regression.regressor import linear_model_fun, lasso_model_fun, ridge_model_fun, svr_model_fun, dt_reg_model_fun, nn_model_fun, rf_reg_model_fun
from time_series_forecasting.forecast import ar_fun, arima_fun, var_fun
from inference.inference import inference_fun, inference_forecast, inference_var_forecast


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
    global model, split_data, metric_val, y_pred

    df = pipeline['data'][0]

    #PRE-CONDITIONS
    if 'combine_dataframe' in pipeline['pre-conditions']:
        df = combine_df_fun(pipeline['data'])
    elif 'combine_generated_data' in pipeline['pre-conditions']:
        df = combine_gen_data_fun(pipeline['data'])

    if 'drop_empty_rows' in pipeline['pre-conditions']:
        df = drop_empty_fun(pipeline['data'][0])
    elif 'fill_empty_rows' in pipeline['pre-conditions']:
        df = fill_empty_fun(pipeline['data'][0])

    if 'one_hot_encoding' in pipeline['pre-conditions']:
        df = one_hot_encode(pipeline['data'][0], pipeline['specific_columns'])
    elif 'label_encoding' in pipeline['pre-conditions']:
        df = label_encode(pipeline['data'][0], pipeline['specific_columns'])

    if 'reshuffle_dataframe' in pipeline['pre-conditions']:
        df = reshuffle_fun(pipeline['data'][0])

    if 'scaling' in pipeline['pre-conditions']:
        df = scaling_fun(pipeline['data'][0], pipeline['scaling_bounds'])
    elif 'scaling_selected_cols' in pipeline['pre-conditions']:
        df = scaling_fun_selected_cols(pipeline['data'][0], pipeline['scaling_bounds'],
                                       pipeline['specific_columns'])

    if 'train_split' in pipeline['pre-conditions']:
        split_data = split_fun(pipeline['data'][0], pipeline['target'], pipeline['test_fraction'])


    #TASKS and MODELS
    #TRAINING
    if 'anomaly_detection' in pipeline['task']:
        if pipeline['model'] == 'autoencoders':
            autoencode_fun(pipeline['data'][0], pipeline['data'][1])
        elif pipeline['model'] == 'isolation_forest':
            isolation_fun(pipeline['data'][0])
        elif pipeline['model'] == 'one_class_svm':
            one_class_svm_fun(pipeline['data'][0])
    elif 'num_classification' in pipeline['task']:
        if pipeline['model'] == 'pac':
            model = pac_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'log':
            model = log_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'mnb':
            model = mnb_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'mlpc':
            model = mlpc_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'svc':
            model = svc_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'dt':
            model = dt_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'kn':
            model = kn_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'rf':
            model = rf_model_fun(split_data[0], split_data[2])
    elif 'text_classification' in pipeline['task']:
        if pipeline['model'] == 'pac':
            model = pac_model_text(split_data[0], split_data[2])
        elif pipeline['model'] == 'log':
            model = log_model_text(split_data[0], split_data[2])
        elif pipeline['model'] == 'mnb':
            model = mnb_model_text(split_data[0], split_data[2])
        elif pipeline['model'] == 'mlpc':
            model = mlpc_model_text(split_data[0], split_data[2])
        elif pipeline['model'] == 'svc':
            model = svc_model_text(split_data[0], split_data[2])
        elif pipeline['model'] == 'dt':
            model = dt_model_text(split_data[0], split_data[2])
        elif pipeline['model'] == 'kn':
            model = kn_model_text(split_data[0], split_data[2])
        elif pipeline['model'] == 'rf':
            model = rf_model_text(split_data[0], split_data[2])
    elif 'text_classification_cross_val' in pipeline['task']:
        if pipeline['model'] == 'pac':
            model = pac_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline['model'] == 'log':
            model = log_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline['model'] == 'mnb':
            model = mnb_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline['model'] == 'mlpc':
            model = mlpc_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline['model'] == 'svc':
            model = svc_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline['model'] == 'dt':
            model = dt_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline['model'] == 'kn':
            model = kn_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])
        elif pipeline['model'] == 'rf':
            model = rf_model_cross_val(split_data[0], split_data[1],
                                        split_data[2], split_data[3])

    elif 'clustering' in pipeline['task']:
        X = df[pipeline['specific_columns'][0]].tolist()
        if pipeline['model'] == 'affinity':
            model = affinity_fun(X)
        elif pipeline['model'] == 'dbscan':
            model = dbscan_fun(X)
        elif pipeline['model'] == 'gmm':
            model = gmm_fun(X)
        elif pipeline['model'] == 'km':
            model = km_fun(X)
        elif pipeline['model'] == 'heirarchy':
            model = heirarchy_fun(X)
        elif pipeline['model'] == 'optics':
            model = optics_fun(X)

    elif 'regression' in pipeline['task']:
        if pipeline['model'] == 'linear':
            model = linear_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'lasso':
            model = lasso_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'ridge':
            model = ridge_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'nn':
            model = nn_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'svr':
            model = svr_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'dt':
            model = dt_reg_model_fun(split_data[0], split_data[2])
        elif pipeline['model'] == 'rf':
            model = rf_reg_model_fun(split_data[0], split_data[2])

    elif 'time_series_forecasting' in pipeline['task']:
        X = df[pipeline['specific_columns'][0]].tolist()
        if pipeline['model'] == 'ar':
            model = ar_fun(X)
        elif pipeline['model'] == 'arima':
            model = arima_fun(X)
        elif pipeline['model'] == 'var':
            model = var_fun(df)

    #INFERENCE
    if 'num_classification' in pipeline['task'] or \
        'text_classification_cross_val' in pipeline['task'] or \
        'text_classification' in pipeline['task'] or \
        'clustering' in pipeline['task']:
        if 'yes' in pipeline['task_description']['inference']['infer']:
            y_pred = inference_fun(split_data[1],
                pipeline['task_description']['inference']['model_name'])

    if 'time_series_forecasting' in pipeline['task']:
        if 'yes' in pipeline['task_description']['inference']['infer']:
            if pipeline['model'] == 'ar' or pipeline['model'] == 'arima':
                y_pred = inference_forecast(split_data[1],
                pipeline['task_description']['inference']['model_name'])
            elif pipeline['model'] == 'var':
                y_pred = inference_var_forecast(split_data[1],
                pipeline['task_description']['inference']['model_name'])


    #POST-CONDITIONS
    if 'save_model' in pipeline['post-conditions']:
        save_model_fun(model, pipeline['model'])

    if 'get_metrics_class' in pipeline['post-conditions']:
        metric_val = get_metrics_class_fun(split_data[1], split_data[3],
                                           model)
    elif 'get_metrics_reg' in pipeline['post-conditions']:
        metric_val = get_metrics_reg_fun(split_data[1], split_data[3],
                                           model)
    elif 'get_metrics_forecast_fun' in pipeline['post-conditions']:
        metric_val = get_metrics_reg_fun(split_data[1], split_data[3],
                                           model)
    elif 'get_metrics_var_forecast_fun' in pipeline['post-conditions']:
        metric_val = get_metrics_var_forecast_fun(split_data[0], split_data[1],
                                         split_data[3], model)




















