from src.paraphrase_helper import extract_qa_from_csv, find_similar_questions_within_faq
import os

# input_file should be stored in faq folder
input_file = "babyBonus.csv"

faq_folder = "./faq"
data_path = os.path.join(faq_folder, input_file)
questions, answers = extract_qa_from_csv(data_path)
assert len(questions) == len(answers)

find_similar_questions_within_faq(questions)