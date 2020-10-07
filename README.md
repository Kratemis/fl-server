# FL-Server

Server for federated learning demo

## Arguments
* Local Folder (Path where local model is generated)
* Config File (Configuration file path for the app)
* Job Id (Unique ID for the process)
* Bucket (S3 Bucket name)
* S3 Access Key
* S3 Secret Key
* S3 Folder
* Models (List of models that are going to be averaged)
* Debug (Debug mode)

## Example
```bash
python main.py --s3-client-models-folder "clients" --s3-main-models-folder "main" --local-client-models-folder "./storage" --client-models "main_model.pt" --config-file "" --job-id 245425 --main-bucket "MY_BUCKET_NAME" --clients-bucket "ANOTHER_BUCKET_NAME"```
