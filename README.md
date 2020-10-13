# FL-Server

Server for federated learning demo

## Arguments
* --s3-client-models-folder (S3 folder for client models)
* --s3-main-models-folder (S3 folder for main models)
* --local-client-models-folder (Local folder for client models)
* --client-models (Comma-separated list of client models to average)
* --job-id (Unique Job ID)
* --clients-bucket (Bucket name for client models)
* --main-bucket (Bucket name for main models)
* --s3-access-key (Credentials for AWS)
* --s3-secret-key (Credentials for AWS)
* --s3-session-tolen (Credentials for AWS)
* --debug (Debug mode)

## Example
```bash
python main.py --s3-client-models-folder "clients" --s3-main-models-folder "main" --local-client-models-folder "./storage" --client-models "main_model.pt" --config-file "" --job-id 245425 --main-bucket "MY_BUCKET_NAME" --clients-bucket "ANOTHER_BUCKET_NAME"```
