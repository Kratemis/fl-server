# FL-Client

Client for federated learning demo

## Arguments
* Download Folder (Path where CIFAR dataset is downloaded)
* Local Folder (Path where local model is generated)
* Config File (Configuration file path for the app)
* Job Id (Unique ID for the process)
* Bucket (S3 Bucket name)
* S3 Access Key
* S3 Secret Key
* S3 Folder
* Main Model Path (Path for the main model)
* Debug (Debug mode)

## Example
```bash
python main.py --download-folder "/tmp" --local-folder "/storage" --config-file "/config/config.json" --job-id 13434 --bucket "911639421134-us-east-1-mybucket" --s3-access-key "1f421f144f14fgt3hth52" --s3-secret-key "4252452g253g532g3gg55g3g35vt4oim35" --main-model-path "/storage/main_model.pt" --debug
```
