import pandas as pd
import pickle
import os
import csv
from app_helper import create_directory_if_missing
import gdown

def download_t5_model_if_not_present(model_path):
    create_directory_if_missing(model_path)
    config_file = 'config.json'
    pytorch_file = 'pytorch_model.bin'

    config_path = os.path.join(model_path, config_file)
    pytorch_path = os.path.join(model_path, pytorch_file)

    if (os.path.isfile(config_path) and os.path.isfile(pytorch_path)):
        print(f"Model found in {model_path}. No downloads needed.")

    else:
        config_location = "https://drive.google.com/uc?id=1x-Y__pyV6pOLn2UOZNNwYXWS0KtL-5WZ"

        pytorch_location = "https://drive.google.com/uc?id=1xvH9-BWqRweJ4Sr1c8oakOMDxI0iTYLd"

        gdown.download(config_location, config_path, False)
        gdown.download(pytorch_location, pytorch_path, False)


def extract_qa_from_csv(path):
    """
    the csv is assumed to have questions on column 1 and answers in column 2
    WITH NO HEADER
    """

    df = pd.read_csv(path, header=None)
    questions = df.iloc[:, 0]
    answers = df.iloc[:, 1]

    return questions, answers


def save_dict(obj, path):
    # Save in .pickle format
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(path):
    # Load .pickle format as dictionary
    with open(path, 'rb') as f:
        return pickle.load(f)


def serialize_to_csv(file_path, input_file, output_file=None):
    """ Convert .pkl file to .csv file & delete legacy .pkl file
    NOTE: input_file should points to a .pkl file. i.e. input_file = "babyBonus" points to "babyBonus.pkl"
    bani_work's generation pipeline always stores the augmented FAQ in .pkl
    To adapt to bani_work's design, I converted the .pkl back to .csv for users' easy viewing
    """

    if output_file is None:
        output_file = input_file
    output_file = output_file + ".csv"
    input_file = input_file + ".pkl"
    input_path = os.path.join(file_path, input_file)
    faq_obj = load_dict(input_path)
    with open(output_file, mode='w') as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writer.writerow(["Question", "Answer", "Label"])
        for faq_unit in faq_obj.FAQ:
            label = faq_unit.question.label
            answer = faq_unit.answer.text
            question = faq_unit.question.text
            csv_writer.writerow([question, answer, label])

    output_path = os.path.join(file_path, output_file)
    # Delete .pkl file
    if os.path.exists(input_path):
        os.remove(input_path)
    os.rename(output_file, output_path)


def store_questions_to_csv(original_questions, questions, output_path, output_file):
    output_file = output_file + ".csv"

    output_path = os.path.join(output_path, output_file)
    with open(output_file, mode='w') as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for question in questions:
            csv_writer.writerow([question, original_questions.index(question)])
    os.rename(output_file, output_path)
