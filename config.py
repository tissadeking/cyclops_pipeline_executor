import yaml

#to retrieve configuration data from the config.yml file

#all files in the docker container are stored inside the folder app
#files_directory = '/app/'
#reads and opens the yml file
yml_file = 'config.yml'
with open(yml_file) as f:
    parameters = yaml.safe_load(f)

#ihu host and port to deploy the container
host = parameters['host']
port = parameters['port']

#MinIO credentials and endpoint for long-term storage
minio_url = parameters['minio_url']  # API port
minio_endpoint = parameters['minio_endpoint']
access_key = parameters['access_key']
secret_key = parameters['secret_key']
bucket_name = parameters['bucket_name']

#to test executor alone or not
test_status = parameters['test_status']
