import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from typing import Any
import argparse
import logging
import boto3
from botocore.exceptions import NoCredentialsError
import time

parser = argparse.ArgumentParser()

parser.add_argument('--s3-client-models-folder', help='S3 folder for client models', required=True)
parser.add_argument('--s3-main-models-folder', help='S3 folder for main models', required=True)
parser.add_argument('--local-folder', help='Local folder', required=True)
parser.add_argument('--client-models', help='Comma-separated list of client models to average', required=True)
parser.add_argument('--job-id', help='Unique Job ID', required=True)
parser.add_argument('--clients-bucket', help='Bucket name for client models', required=True)
parser.add_argument('--main-bucket', help='Bucket name for main models', required=True)
parser.add_argument('--s3-access-key', help='Credentials for AWS', required=False)
parser.add_argument('--s3-secret-key', help='Credentials for AWS',  required=False)
parser.add_argument('--s3-session-token', help='Credentials for AWS', required=False)
parser.add_argument('-d', '--debug', help="Debug mode for the script")
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


device = torch.device("cuda:0" if config['config']['use_cuda'] else "cpu")

logging.info("DEVICE: ")
logging.info(device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def key_average(list, key):
    try:
        average = 0

        for item in list:
            logging.debug("item[key] " + str(item[key]))
            logging.debug("average: " + str(average))
            average = average + item[key]
    except Exception as e:
        logging.error('Exception in key_average')
        logging.error(e)
        exit(1)
    logging.debug('FINAL AVERAGE')
    logging.debug(average / len(list))
    return "{:e}".format(average / len(list))


def add_model(dst_model, src_model):
    """Add the parameters of two models.

    Args:
        dst_model (torch.nn.Module): the model to which the src_model will be added.
        src_model (torch.nn.Module): the model to be added to dst_model.
    Returns:
        torch.nn.Module: the resulting model of the addition.

    """

    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data + dict_params2[name1].data)
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.

    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.

    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def federated_avg(models: Dict[Any, torch.nn.Module]) -> torch.nn.Module:
    """Calculate the federated average of a dictionary containing models.
       The models are extracted from the dictionary
       via the models.values() command.

    Args:
        models (Dict[Any, torch.nn.Module]): a dictionary of models
        for which the federated average is calculated.

    Returns:
        torch.nn.Module: the module with averaged parameters.
    """
    nr_models = len(models)
    model_list = list(models.values())
    model = type(model_list[0])()
    for p in model.parameters():
        p.data.fill_(0)

    for i in range(nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)

    return model


def upload_to_aws(local_file, bucket, s3_file):
    logging.info("Uploading to S3 bucket ")

    logging.debug("Local File: " + local_file)
    logging.debug("Bucket: " + bucket)
    logging.debug("S3 File: " + s3_file)
    s3 = boto3.client('s3', aws_access_key_id=args.s3_access_key,
                      aws_secret_access_key=args.s3_secret_key,
                      aws_session_token=args.s3_session_token)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        logging.info("Upload Successful")
        return True
    except FileNotFoundError:
        logging.error("The file was not found")
        return False
    except NoCredentialsError:
        logging.error("Credentials not available")
        return False


def download_from_aws(bucket, remote_path, local_path):
    logging.info("Downloading from S3 bucket")

    s3 = boto3.client('s3', aws_access_key_id=args.s3_access_key,
                      aws_secret_access_key=args.s3_secret_key,
                      aws_session_token=args.s3_session_token)

    try:
        logging.debug("Bucket: " + bucket)
        logging.debug("Remote Path: " + remote_path)
        logging.debug("Local Path: " + local_path)
        s3.download_file(bucket, remote_path, local_path)
        logging.info("Download Successful")
        return True
    except FileNotFoundError:
        logging.error("The file was not found")
        return False
    except NoCredentialsError:
        logging.error("Credentials not available")
        return False


# Setup
models = []
state_dicts = []
try:
    logging.info("Loading all models")

    dir_items = args.client_models.split(',')

    logging.debug("Models found:")
    logging.debug(dir_items)

    counter = 0
    for item in dir_items:
        logging.debug("Complete remote path: " + args.local_folder + "/" + item)
        download_from_aws(args.clients_bucket, args.s3_client_models_folder + "/" + item, args.local_folder + "/" + item)
        # Checksum check here?
        logging.debug("Loading model...")
        model = torch.load(args.local_folder + "/" + item)
        model.to(device) #GPU
        logging.debug("Appending model to models array")
        models.append(model)
        logging.debug("Get state dict of the model...")
        sd = model.state_dict()
        logging.debug("Appending model to state_dicts array")
        state_dicts.append(sd)

        # torch.save(model.state_dict(), args.local_folder + "/model_" + str(counter) + '.pt')
        counter = counter + 1

except Exception as e:
    logging.error(e)
    exit(1)

averages = {}

logging.info("Averaging " + str(len(models)) + "...")

models_dict = {i: models[i] for i in range(0, len(models))}

logging.debug("Models dict: ")
logging.debug(models_dict)

federated_model = federated_avg(models_dict)
federated_model.to(device)

FINAL_MODEL_NAME = 'main_model.pt'
FINAL_MODEL_PATH = args.local_folder + '/' + FINAL_MODEL_NAME

torch.save(federated_model, FINAL_MODEL_PATH)
uploaded = upload_to_aws(FINAL_MODEL_PATH, args.main_bucket,
                         args.s3_main_models_folder + '/' + str(int(time.time())) + '_' + FINAL_MODEL_NAME)

logging.info("Model Saved")
