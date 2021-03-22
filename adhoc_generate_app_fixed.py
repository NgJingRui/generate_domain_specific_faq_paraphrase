import os

from t5_paraphrase import T5Generator

input_question = "Who do I contact for enquiries and feedback on Approved Institutions (AI)?"

generate_n_paraphrases = 40
keep_top_k_paraphrases = 11
assert keep_top_k_paraphrases <= generate_n_paraphrases, "keep_top_k_paraphrases should be <= generate_n_paraphrases"

# input_file = None if you do not have any supporting FAQ (where the input_question is from)
input_file = "babyBonus.csv"

qqp_producer = T5Generator(model_path="./models/t5_qqp/",
                           top_p=0.98, num_return=generate_n_paraphrases, max_len=128, top_k=120, is_early_stopping=True)

candidate_paraphrases_w_score_position = qqp_producer.adhoc_generate(input_question, generate_n_paraphrases,
                                                                     keep_top_k_paraphrases, input_file)
print(f"Original:\n{input_question}\n")
count = 1
print(f"Paraphrases:")
for pcp_tuple in candidate_paraphrases_w_score_position:
    print(f"{count}. {pcp_tuple[0]} ({pcp_tuple[1]}, {pcp_tuple[2]})\n")
    count += 1