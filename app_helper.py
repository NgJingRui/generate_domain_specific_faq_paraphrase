import os


def set_up_excluded_folders():
    output_folder = "outputs"
    model_folder = "models"

    create_directory_if_missing(output_folder)
    create_directory_if_missing(model_folder)


def create_directory_if_missing(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def path_to_folder_file(path):
    tokens = path.split("/")
    input_folder = ""
    input_file = ""
    last_ind = len(tokens) - 1
    for i, token in enumerate(tokens):
        if i != last_ind:
            input_folder = os.path.join(input_folder, token)
        else:
            input_file = token
    return input_folder, input_file


def handle_output_file_variations(output_file):
    # Handle variations such as "babyBonus.csv" and "babyBonus" -> return "babyBonus"
    if len(output_file.split(".")) > 1:
        return output_file.split(".")[0]
    else:
        return output_file


def handle_path_or_no_path(input_folder, input_arg):
    if len(input_arg.split("/")) > 1:
        # PATH GIVEN IN ARGUMENT
        input_folder, input_file = path_to_folder_file(input_arg)
    else:
        # NO PATH GIVEN IN ARGUMENT
        input_file = input_arg
    return input_folder, input_file
