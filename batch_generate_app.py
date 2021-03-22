import os, sys, getopt
from src.app_helper import handle_path_or_no_path, handle_output_file_variations
from src.bani_work.FAQ import FAQ
from src.bani_work.generation import GenerateManager
from src.t5_paraphrase import T5Generator
from src.paraphrase_helper import extract_qa_from_csv, serialize_to_csv, store_questions_to_csv
from src.app_helper import set_up_excluded_folders, create_directory_if_missing

# Compulsory Arguments
input_file = ""
faq_name = ""

# Optional Arguments & their Default Values
input_folder = "./faq"
output_folder = sys.argv[0].split(".")[0]
generate_n_paraphrases = 10
keep_top_k_paraphrases = 5
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:nk", ["ifile=", "ofile=", "n_gen=", "k_top="])
except getopt.GetoptError:
    print("batch_generate_app.py -i <inputfile> -o <outputfile> "
          "--n_gen <generate n paraphrases> --k_top <retain top k paraphrases>")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-i", "--ifile"):
        # opt can be the input file (must be stored in ./faq)
        # or the input path (includes the file name) -> (input file can be stored in other folder)
        input_folder, input_file = handle_path_or_no_path(input_folder, arg)
        assert input_file.split(".")[-1] == "csv", \
            "Input file should have a '.csv' extension as only .csv input files are currently supported by the system"
    elif opt in ("-o", "--ofile"):
        output_folder, faq_name = handle_path_or_no_path(output_folder, arg)
        faq_name = handle_output_file_variations(faq_name)
    elif opt in ("-n", "--n_gen"):
        generate_n_paraphrases = int(arg)
        assert type(generate_n_paraphrases) == int, "generate_n_paraphrases should be an integer"
        assert generate_n_paraphrases > 0, "generate_n_paraphrases should be positive"
    elif opt in ("-k", "--k_top"):
        keep_top_k_paraphrases = int(arg)
        assert type(keep_top_k_paraphrases) == int, "keep_top_k_paraphrases should be an integer"
        assert keep_top_k_paraphrases > 0, "keep_top_k_paraphrases should be positive"
        assert keep_top_k_paraphrases <= generate_n_paraphrases, "keep_top_k_paraphrases should be <= generate_n_paraphrases"

# 0. Set up folders excluded in .gitignore and folder to store the augmented faq (specified by faq_store)
set_up_excluded_folders()

faq_store = os.path.join("outputs", output_folder)
create_directory_if_missing(faq_store)

# 1. Load Questions and Answers from FAQ in .csv format
data_path = os.path.join(input_folder, input_file)
questions, answers = extract_qa_from_csv(data_path)
assert len(questions) == len(answers)
print(f"Succesfully retrieved questions & answers from {data_path}")
print(f"Augmented FAQ will be stored to {faq_store}.")
# 2. Preparations before generating paraphrases of the questions
qqp_producer = T5Generator(model_path="./models/t5_qqp/",
                           top_p=0.98, num_return=generate_n_paraphrases, max_len=128, top_k=120, is_early_stopping=True)

names = ["T5"]
quantity = [keep_top_k_paraphrases]

FAQ_obj = FAQ(name=faq_name, questions=questions[280:], answers=answers[280:])

generatorManager = GenerateManager(
    producers=[
        qqp_producer
    ],
    names=names,
    nums=quantity,
)

# 3. Generate paraphrases of the questions
FAQ_obj.buildFAQ(generatorManager)

FAQ_obj.save(faq_store)

print(f"Augmented FAQ ({faq_name}.pkl) has been saved to {faq_store}")

# 4. Post-generation to format outputs for easier viewing by users
serialize_to_csv(faq_store, faq_name)
store_questions_to_csv(qqp_producer.original_sentences, qqp_producer.inconsistent_sentences, faq_store, output_file="inconsistent_sentences")
store_questions_to_csv(qqp_producer.original_sentences, qqp_producer.not_generated_sentences, faq_store, output_file="not_generated_sentences")

# 5. Inform users of the results of the generation run
inconsistent_count = len(qqp_producer.inconsistent_sentences)
not_generated_count = len(qqp_producer.not_generated_sentences)
total_count = len(qqp_producer.original_sentences)

if not_generated_count == 0:
    path = os.path.join(faq_store, faq_name + ".csv")
    print(f"We have successfully generated paraphrases for all {total_count} questions.")
    print(f"You may view your augmented FAQ at {path}\n")
else:
    path = os.path.join(faq_store,"not_generated_sentences" + ".csv")
    print(f"All generated paraphrases of {not_generated_count} questions did not pass through the candidate paraphrase selection phase.")
    print(f"You may view these questions at {path}\n")
if inconsistent_count != 0:
    path = os.path.join(faq_store,"inconsistent_sentences.csv")
    print(f"We have found {inconsistent_count} questions with inconsistent usage of domain-specific terms, but went ahead with generating their paraphrases.")
    print(f"You may view these questions at {path}\n")
