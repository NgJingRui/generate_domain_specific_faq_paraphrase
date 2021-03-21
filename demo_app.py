import os, sys

# Important Variables
input_folder = "./faq"
input_file = "babyBonus.csv"
faq_name = "babyBonus"
generate_n_paraphrases = 10
keep_top_k_paraphrases = 5

# output_folder is the script name by default
output_folder = sys.argv[0].split(".")[0]
output_path = os.path.join(output_folder, faq_name)
print(f"python3 batch_generate_app.py -i {input_file} -o {output_path} --n_gen {generate_n_paraphrases} --k_top {keep_top_k_paraphrases}")
os.system(f"python3 batch_generate_app.py -i {input_file} -o {output_path} --n_gen {generate_n_paraphrases} --k_top {keep_top_k_paraphrases}")