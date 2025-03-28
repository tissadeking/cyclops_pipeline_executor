import requests
import json

#script to send intents

#API endpoint for sending intents
url = "http://127.0.0.1:5003/pipeline"

pipeline = {
            "intent_type": "analytical",
            "data": ["policy_store", "dataframe 2", "...."],
            "task": "text_classification",
            "task_description": {
                "training": "yes",
                "inference": {
                    "infer": "no",
                    "model_name": "pac.sav",
                },
                "save_model": "yes"
            },
            "model": "mnb",
            "target": "data",
            #"specific_columns": ["feature 1", "feature 2", "feature 3"],
            "specific_columns": [],
            "test_fraction": 0.2,
            "pre-conditions": ["combine_text_data", "extract_x_y_text",
            "train_split"],
            "scaling_bounds": [-1, 1],
            "post-conditions": ["save_model", "get_metrics_class"],
            "pipeline_id": "XYTHSI33"
        }
'''      
pipeline = {
            "intent_type": "analytical",
            "data": ["combined_2", "dataframe 2", "...."],
            "task": "num_classification",
            "task_description": {
                "training": "yes",
                "inference": {
                    "infer": "no",
                    "model_name": "log.sav",
                },
                "save_model": "yes"
            },
            "model": "log",
            "target": "drift",
            #"specific_columns": ["feature 1", "feature 2", "feature 3"],
            "specific_columns": [],
            "test_fraction": 0.2,
            "pre-conditions": ["encode_target", "convert_np_array",
            "train_split"],
            #"pre-conditions": ["extract_x_y", "train_split"],
            "scaling_bounds": [-1, 1],
            "post-conditions": ["save_model", "get_metrics_class"],
            "pipeline_id": "XYTHSI33"
        }

pipeline = {
            "intent_type": "analytical",
            "data": ["combined_2", "dataframe 2", "...."],
            "task": "regression",
            "task_description": {
                "training": "yes",
                "inference": {
                    "infer": "no",
                    "model_name": "linear.sav",
                },
                "save_model": "yes"
            },
            "model": "linear",
            "target": "drift",
            #"specific_columns": ["feature 1", "feature 2", "feature 3"],
            "specific_columns": [],
            "test_fraction": 0.2,
            #"pre-conditions": ["encode_target", "convert_np_array",
            #"train_split"],
            "pre-conditions": ["extract_x_y", "train_split"],
            "scaling_bounds": [-1, 1],
            "post-conditions": ["save_model", "get_metrics_reg"],
            "pipeline_id": "XYTHSI33"
        }
'''

# Convert dictionary to JSON
headers = {"Content-Type": "application/json"}
response = requests.put(url, data=json.dumps(pipeline), headers=headers, timeout=10)
#print(response)
# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json())
