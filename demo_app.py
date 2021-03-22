import os, sys

# Important Arguments
input_folder = "./faq"
input_file = "babyBonus.csv"
faq_name = "babyBonus"
generate_n_paraphrases = 10
keep_top_k_paraphrases = 5
assert keep_top_k_paraphrases <= generate_n_paraphrases, "keep_top_k_paraphrases should be <= generate_n_paraphrases"

# output_folder is the script name by default
output_folder = sys.argv[0].split(".")[0]
output_path = os.path.join(output_folder, faq_name)

# Mac users may have to update "python" to "python3". Python version used: 3.7.7
print(f"python batch_generate_app.py -i {input_file} -o {output_path} --n_gen {generate_n_paraphrases} --k_top {keep_top_k_paraphrases}")
os.system(f"python batch_generate_app.py -i {input_file} -o {output_path} --n_gen {generate_n_paraphrases} --k_top {keep_top_k_paraphrases}")