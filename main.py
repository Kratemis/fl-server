import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from typing import Any
import logging
import boto3
from botocore.exceptions import NoCredentialsError
import os
import json


def load_config():
    return json.loads(str(os.environ['CONFIG']))


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
    s3 = boto3.client('s3')

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

    s3 = boto3.client('s3')

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


config = load_config()

if config['metadata']['debug']:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

for local_model_path in [config['input_model']['local_path'], config['output']['local_path']]:
    if not os.path.exists(local_model_path):
        logging.info("Creating directory")
        os.makedirs(local_model_path)

models = []
state_dicts = []
try:
    logging.info("Loading all models")

    dir_items = config['input_models']['model_list'].split(',')

    logging.debug("Models found:")
    logging.debug(dir_items)

    counter = 0
    for item in dir_items:
        logging.debug("Complete remote path: " + config['input_models']['local_path'] + "/" + item)
        download_from_aws(config['input_models']['s3_s3_bucket'], config['input_models']['s3_key'],
                          config['input_models']['local_path'] + "/" + item)
        # Checksum check here?
        logging.debug("Loading model...")
        model = torch.load(config['input_models']['local_path'] + "/" + item)
        logging.debug("Appending model to models array")
        models.append(model)
        logging.debug("Get state dict of the model...")
        sd = model.state_dict()
        logging.debug("Appending model to state_dicts array")
        state_dicts.append(sd)

        # torch.save(model.state_dict(), config['input_models']['local_path'] + "/model_" + str(counter) + '.pt')
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

FINAL_MODEL_NAME = 'main_model.pt'
FINAL_MODEL_PATH = config['input_models']['local_path'] + '/' + FINAL_MODEL_NAME

torch.save(federated_model, FINAL_MODEL_PATH)
uploaded = upload_to_aws(FINAL_MODEL_PATH, config['output']['s3_bucket'],
                         config['output']['s3_key'])

logging.info("Model Saved")
