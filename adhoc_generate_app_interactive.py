from src.t5_paraphrase import T5Generator

input_question = str(input("Generate paraphrases for this question: "))

generate_n_paraphrases = int(input("# of paraphrase to generate for given question, e.g. 10: "))
assert generate_n_paraphrases > 0, "generate_n_paraphrases should be positive"


keep_top_k_paraphrases = int(input("# of top paraphrases to retain from candidate selection process, e.g. 5: "))
assert keep_top_k_paraphrases > 0, "keep_top_k_paraphrases should be positive"

input_file = str(input("Name of FAQ file to retrieve questions to compare with generated paraphrases, e.g. babyBonus.csv: "))

# Adhoc-Generation without any supporting FAQ
if input_file == "-1":
    input_file = None
    print("YES!")
if input_file is not None:
    assert input_file.split(".")[-1] == "csv", \
                "Input file should have a '.csv' extension as only .csv input files are currently supported by the system"

qqp_producer = T5Generator(model_path="./models/t5_qqp/",
                           top_p=0.98, num_return=generate_n_paraphrases, max_len=128, top_k=120, is_early_stopping=True)

candidate_paraphrases_w_score_position = qqp_producer.adhoc_generate(input_question, generate_n_paraphrases,
                                                                     keep_top_k_paraphrases, input_file)
print(f"Original:\n{input_question}\n")
count = 1
print(f"Paraphrases:\n")
for pcp_tuple in candidate_paraphrases_w_score_position:
    print(f"{count}. {pcp_tuple[0]} ({pcp_tuple[1]}, {pcp_tuple[2]})\n")
    count += 1