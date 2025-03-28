from minio_crud import retrieve_object, create_object

def add_objects():
    #add these objects to the minio storage
    create_object('policy_store', 'csv/policy_store.csv')
    create_object('combined_2', 'csv/combined_2.csv')
