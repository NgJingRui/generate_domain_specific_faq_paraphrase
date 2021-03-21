from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

from abbreviation_helper import get_abbreviation_dict, remove_abbreviation_expansion, reinstate_abbreviation_expansion, check_inconsistent
from bani_work.rajat_work.base import BaseGenerator


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class T5Generator(BaseGenerator):
    """ Generate questions using a T5 Model Trained on Paraphrase Dataset -
    QQP (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
    """

    def __init__(self,
                 model_path="ramsrigouthamg/t5_paraphraser",
                 top_p=0.98, num_return=11, max_len=128, top_k=120, is_early_stopping=True,
                 can_model_path="paraphrase-distilroberta-base-v1"):
        """
        Load model with model path and initialize parameters for generation
        :param str model_path: Points to the directory where the model's config.json and pytorch_model.bin is stored.
                               Can also point to a path where it loads a model from the web.
        :param float top_p: only the most probable tokens with probabilities that add up to top_p or higher are kept for
                            generation.
        :param int num_return: Number of sentences to return from the generate() method
        :param int max_len: Maximum length that will be used in the generate() method
        :param int top_k: Number of highest probability vocabulary tokens to keep for top-k-filtering in the generate()
                          method
        :param bool is_early_stopping: Flag that will be used to determine whether to stop the beam search
        """
        super().__init__("T5 Model - Paraphrase Generation")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)

        # Parameters for generation
        self.top_p = top_p
        self.num_return = num_return
        self.max_len = max_len
        self.top_k = top_k
        self.is_early_stopping = is_early_stopping

        # Load for candidate paraphrase selection
        self.can_model = SentenceTransformer(can_model_path)
        self.original_sentences = []

        self.inconsistent_sentences = []
        self.not_generated_sentences = []

    def _load_model(self, model_path):
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model = model.to(self.device)
        return model

    @staticmethod
    def _preprocess(sentence, abbrevs):
        sentence = remove_abbreviation_expansion(sentence, abbrevs)
        return sentence

    @staticmethod
    def _post_process(sentence, abbrevs):
        sentence = reinstate_abbreviation_expansion(sentence, abbrevs)
        return sentence

    def generate(self, sentence):
        """
        Generate paraphrases for a given sentence

        :param str sentence: Original sentence used for generating its paraphrases

        """
        text = "paraphrase: " + sentence + "</s>"
        encoding = self.tokenizer.encode_plus(text, padding='max_length', return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            max_length=self.max_len,
            top_k=self.top_k,
            top_p=self.top_p,
            early_stopping=self.is_early_stopping,
            num_return_sequences=self.num_return
        )

        sentences_generated = []
        # sentence = re.sub(r'[^\w\s]', '', sentence)
        for output in outputs:
            sent = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in sentences_generated:
                sentences_generated.append(sent)

        return sentences_generated

    def batch_generate(self, sentences):
        # For use in candidate_selection method
        self.original_sentences = sentences
        # Reset variables storing inconsistent and not generated sentences
        self.inconsistent_sentences = []
        self.not_generated_sentences = []

        results = dict()
        # seen_questions used to clean up duplicates across and within label. Each label points to a unique FAQ pair.
        seen_questions = [sentence for sentence in sentences]
        for sentence in tqdm(sentences):
            if not check_inconsistent(sentence):
                sentences_generated = self.generate_with_processing(sentence)
            else:
                self.inconsistent_sentences.append(sentence)
                sentences_generated = self.generate(sentence)

            # 4a. Preparation for Candidate Paraphrase Selection
            similarity_scores = [self._similarity_score(sentence, paraphrase) for paraphrase in sentences_generated]
            positions = self._get_positions(sentence, sentences_generated)

            assert len(similarity_scores) == len(sentences_generated)
            assert len(positions) == len(sentences_generated)

            # 4b. Passing paraphrase through 2-step selection process & Sort by Descending Score
            paraphrase_score_tuples = []
            for paraphrase, score, position in zip(sentences_generated, similarity_scores, positions):
                if paraphrase not in seen_questions:
                    # 2-Step Selection Process
                    if float(score) >= 4.0 and int(position) in [1]:
                        paraphrase_score_tuples.append((paraphrase, score, position))
                    seen_questions.append(paraphrase)

            # Sort by descending score
            paraphrase_score_tuples.sort(key=lambda x: x[1], reverse=True)
            candidate_paraphrases = [pst_tuple[0] for pst_tuple in paraphrase_score_tuples]

            # results[sentence] = sentences_generated
            results[sentence] = candidate_paraphrases
            if len(candidate_paraphrases) == 0:
                self.not_generated_sentences.append(sentence)
        return results

    def generate_with_processing(self, sentence):
        abbrevs = get_abbreviation_dict(sentence)
        new_abbrevs = abbrevs
        # new_abbrevs = {key + "_1": value for key, value in abbrevs.items()}
        # 1. Pre Processing
        sentence_ready = self._preprocess(sentence, new_abbrevs)

        # 2. Generation
        sentences_generated = self.generate(sentence_ready)

        # 3. Post Processing
        sentence_return = []
        for sentence_gen in sentences_generated:
            sentence_return.append(self._post_process(sentence_gen, new_abbrevs))

        return sentence_return

    def _candidate_selection(self, original, generated_paraphrases,
                             lower_bound=4.0, position_choices=[1]):

        # Step 1: Filter by Similarity Scores
        similarity_scores = [self._similarity_score(original, paraphrase) for paraphrase in generated_paraphrases]
        positions = self._get_positions(original, generated_paraphrases)

        assert len(similarity_scores) == len(generated_paraphrases)
        assert len(positions) == len(generated_paraphrases)

        # 4b. Passing paraphrase through 2-step selection process & Sort by Descending Score
        paraphrase_score_tuples = []
        for paraphrase, score, position in zip(generated_paraphrases, similarity_scores, positions):
            if float(score) >= float(lower_bound) and int(position) in position_choices:
                paraphrase_score_tuples.append((paraphrase, score, position))

        # Sort by descending score
        paraphrase_score_tuples.sort(key=lambda x: x[1], reverse=True)
        candidate_paraphrases = [pst_tuple[0] for pst_tuple in paraphrase_score_tuples]

        return candidate_paraphrases

    def _similarity_score(self, original, paraphrase):
        sentences1 = [original]
        sentences2 = [paraphrase]

        # Compute embedding for both lists
        embeddings1 = self.can_model.encode(sentences1, convert_to_tensor=True, show_progress_bar=False)
        embeddings2 = self.can_model.encode(sentences2, convert_to_tensor=True, show_progress_bar=False)

        # Compute cosine-similarities
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        score = cosine_scores[0][0] * 5
        return round(float(score), 2)

    def _get_positions(self, original, candidate_paraphrases):
        corpus = self.original_sentences
        corpus_embeddings = self.can_model.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
        positions = []
        for paraphrase in candidate_paraphrases:
            query = [paraphrase]
            top_k = min(5, len(corpus))
            query_embedding = self.can_model.encode(query, convert_to_tensor=True, show_progress_bar=False)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            corpus_matches = [corpus[idx] for idx in top_results[1]]
            positions.append(self.check_position(corpus_matches, original))

        return positions

    @staticmethod
    def check_position(corpus_matches, original):
        # Return 1,2,3,4,5 for their respective position of the query_original_q in corpus_matches
        # Return -1 if not found
        for ind, match in enumerate(corpus_matches):
            if match.lower() == original.lower():
                return ind + 1
        return -1

    def debug_generate(self, sentence, is_preprocess=True, is_postprocess=True,
                       is_select=True, lower_bound=4.0, position_choices=[1]):
        sentence_ready = sentence

        print(f"Generating for: {sentence_ready}")
        if is_preprocess:
            abbrevs = get_abbreviation_dict(sentence)
            # new_abbrevs = {key + "_1": value for key, value in abbrevs.items()}
            print(f"is_preprocess=True: \nabbrevs:{abbrevs},\nnew_abbrevs:{abbrevs}\n")
            sentence_ready = self._preprocess(sentence, abbrevs)
            print(f"After pre-processing, it becomes: {sentence_ready}")

        print(f"Generating for: {sentence_ready}")
        sentences_generated = self.generate(sentence_ready)

        sentence_return = []
        if is_postprocess:
            for sentence_gen in sentences_generated:
                processed_sentence = self._post_process(sentence_gen, abbrevs)
                sentence_return.append(processed_sentence)
                print(f"Before: {sentence_gen}\nAfter: {processed_sentence}")
                print("\n")
        else:
            sentence_return = sentences_generated

        res = []
        if is_select:
            if not self.original_sentences:
                self.original_sentences = [sentence]
            res = self._candidate_selection(sentence, sentence_return, lower_bound, position_choices)
        else:
            res = sentence_return
        return res

    def _save_intermediate_output(self, saveAll, outputPath, outputName):
        import csv
        import os
        with open(outputName + ".csv", mode='w') as new_csv:
            csv_writer = csv.writer(new_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in saveAll:
                csv_writer.writerow([row[0], row[1], row[2], row[3]])

        os.rename(outputName + ".csv", os.path.join(outputPath, outputName + ".csv"))

