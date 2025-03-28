from flask import Flask, request, jsonify
from execute import execute_fun
from minio_crud import ensure_bucket
import config, add_files

parameters = config.parameters
host = config.host
port = config.port
test_status = config.test_status
print("Executor has started!", flush=True)

if test_status == "yes":
    #create long term storage bucket and add files into it for testing
    ensure_bucket()
    add_files.add_objects()


app = Flask(__name__, static_folder="static")

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

#global field_data

# GET: Retrieve data
@app.route('/pipeline', methods=['GET'])
def get_fields():
    try:
        #if field_data:
        return jsonify(field_data), 200
    except:
        return jsonify(pipeline), 200

#PUT: Update data
@app.route('/pipeline', methods=['PUT'])
def update_field():
    global field_data
    data = request.get_json()
    field_data = data
    execute_fun(field_data)
    return jsonify({"message": "Pipeline sent", "data": field_data}), 200


#if __name__ == "__main__":
#    app.run(port=5002, debug=True)
if __name__ == "__main__":
    app.run(host=host, port=port)
