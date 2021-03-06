import os
print("You are currently performing batch generate of paraphrases for all questions from a FAQ file")
input_file = input("Name of the FAQ file or its relative path from current working directory: ")

assert input_file.split(".")[-1] == "csv", \
            "Input file should have a '.csv' extension as only .csv input files are currently supported by the system"

output_file = input("Name of output file or its relative path from .outputs folder: ")

generate_n_paraphrases = int(input("# of paraphrase to generate for each question: "))
assert generate_n_paraphrases > 0, "generate_n_paraphrases should be positive"

keep_top_k_paraphrases = int(input("# of top paraphrases to retain from candidate selection process: "))
assert keep_top_k_paraphrases > 0, "keep_top_k_paraphrases should be positive"

assert keep_top_k_paraphrases <= generate_n_paraphrases, "keep_top_k_paraphrases should be <= generate_n_paraphrases"

# Mac users may have to update "python" to "python3". Python version used: 3.7.7
print(f"python batch_generate_app.py -i {input_file} -o {output_file} --n_gen {generate_n_paraphrases} --k_top {keep_top_k_paraphrases}")
os.system(f"python batch_generate_app.py -i {input_file} -o {output_file} --n_gen {generate_n_paraphrases} --k_top {keep_top_k_paraphrases}")
print("DONE!")