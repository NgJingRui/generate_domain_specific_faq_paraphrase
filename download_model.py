from app_helper import create_directory_if_missing
import os
import gdown

model_folder = "t5_qqp"
model_path = os.path.join("./models", "t5_qqp")
create_directory_if_missing(model_path)

config_file = 'config.json'
config_path = os.path.join(model_path, config_file)
config_location = "https://drive.google.com/uc?id=1x-Y__pyV6pOLn2UOZNNwYXWS0KtL-5WZ"

model_file = 'pytorch_model.bin'
model_path = os.path.join(model_path, model_file)
model_location = "https://drive.google.com/uc?id=1xvH9-BWqRweJ4Sr1c8oakOMDxI0iTYLd"

if os.path.isfile(config_path) and os.path.isfile(model_path):
    print(f"Model found in {model_path}. No downloads needed.")
else:
    gdown.download(config_location, config_path, False)
    gdown.download(model_location, model_path, False)

