import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# resource:
# https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights

def download_and_load_gpt2(model_size, models_dir):
    # validate model size
    allowed_sizes = ['124M', '355M', '774M', '1558M']
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")
    
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe"
    ]

    # os.makedirs(model_dir, exist_ok=True)
    # for filename in filenames:
    #     file_url = os.path.join(base_url, model_size, filename)
    #     file_path = os.path.join(model_dir, filename)
    #     downdload_file(file_url, file_path)

    # load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, 'hparams.json')))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def downdload_file(url, destination):
    try:
        response = requests.get(url, stream=True, verify=False)
        
        # get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get('Content-Length', 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"file already exists and is up to date: {destination}")
                return
        
        block_size = 1024  # 1 KB

        progress_bar_description = url.split('/')[-1] # extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings['n_layer'])]}

    # interate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # process the variable name to extract relevant parts
        varialbe_name_parts = name.split('/')[1:] # skip the 'model/' prefix

        # identify the target dictionary for the variable
        target_dict = params
        if varialbe_name_parts[0].startswith('h'):
            layer_number = int(varialbe_name_parts[0][1:])
            target_dict = params['blocks'][layer_number]

        # recursively access or create nested dictionaries
        for key in varialbe_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # assign the variable array to the last key
        last_key = varialbe_name_parts[-1]
        target_dict[last_key] = variable_array
    

    return params
