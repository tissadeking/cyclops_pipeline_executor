






pipeline = {
            "intent_type": "",
            "data": [],
            "task": "",
            "task_description": {
                "training": "",
                "inference": {
                    "infer": "",
                    "model_name": ""
                },
                "save_model": ""
            },
            "model": "",
            "target": "",
            "specific_columns": [],
            "test_fraction": 0.2,
            "scaling_bounds": [-1, 1],
            "pre-conditions": [],
            "post-conditions": [],
            "pipeline_id": ""
        }

#eg
pipeline_2 = {
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
        }
