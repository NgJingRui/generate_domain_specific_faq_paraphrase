from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize

from src.abbreviation import Abbreviation
from abbreviations import schwartz_hearst


def get_abbreviation_dict(sentences):
    if type(sentences) is list:
        abbreviation_dict = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=" ".join(sentences),
                                                                                  most_common_definition=True)
    elif type(sentences) is str:
        abbreviation_dict = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=sentences,
                                                                                  most_common_definition=True)
    return abbreviation_dict


def find_all_abbreviations(sentence, abbrevs):
    tokens_text = word_tokenize(sentence)
    abbrev_data = []
    for abbrev in abbrevs.keys():
        indices = [i for i, token in enumerate(tokens_text) if token == abbrev]
        for index in indices:
            abbrev_data.append(Abbreviation(abbrev, abbrevs[abbrev], index))

    return abbrev_data


def remove_abbreviation_expansion(sentence, abbrevs):
    tokens_text = word_tokenize(sentence)
    abbrev_data = find_all_abbreviations(sentence, abbrevs)
    if len(abbrev_data) == 0:
        return sentence
    start = 0
    end = len(tokens_text) - 1

    retain_portions = []
    result_tokens = []

    abbrev_data.sort(key=lambda x: x.start_index)
    # Store parts of the tokens_text that does not have the abbreviation and its expansion in "retain_portions"
    for abbrev in abbrev_data:
        retain_portions.append(tokens_text[start: abbrev.start_index])
        # Update start_index
        start = abbrev.end_index + 1

    # Edge case: portion of tokens_text after the last abbreviation's end index
    if abbrev_data[-1].end_index != end:
        last_start = abbrev_data[-1].end_index + 1
        retain_portions.append(tokens_text[last_start:])
    # print(retain_portions)

    # Combine retain_portions and their respective abbreviation in "result_tokens"
    for ind, abbrev in enumerate(abbrev_data):
        result_tokens.extend(retain_portions[ind])
        result_tokens.append(abbrev.abbrev)
    if len(retain_portions) != len(abbrev_data):
        result_tokens.extend(retain_portions[-1])

    return TreebankWordDetokenizer().detokenize(result_tokens)


def reinstate_abbreviation_expansion(sentence, abbrevs):
    # abbrevs = {key + "_1": value for key, value in abbrevs.items()}
    tokens_text = word_tokenize(sentence)
    abbrev_data = find_all_abbreviations(sentence, abbrevs)
    if len(abbrev_data) == 0:
        return sentence
    abbrev_data.sort(key=lambda x: x.index)

    result_tokens = []
    start = 0
    for abbrev in abbrev_data:
        start_tokens = tokens_text[start:abbrev.index]
        start_tokens.append(abbrev.construct_abbrev_with_expansion())
        result_tokens.extend(start_tokens)
        start = abbrev.index + 1
    result_tokens.extend(tokens_text[start:])

    return TreebankWordDetokenizer().detokenize(result_tokens)


def check_inconsistent(question):
    """
    This method checks for inconsistent usage of domain-specific terms
    For example, "My spouse is the nominated CDA Trustee and I have applied for the Baby Bonus Scheme.
                  How can I open the Child Development Account (CDA) for my child?"
    -> CDA inconsistently referred to using Child Development Account (CDA) and CDA
    """
    abbrev_dict = get_abbreviation_dict(question)
    question_tokens = word_tokenize(question)

    if abbrev_dict:
        for abbrev in abbrev_dict.keys():
            abbrev_str = "(" + abbrev + ")"
            abbrev_str_count = question.count(abbrev_str)
            count = question_tokens.count(abbrev)

            if count != abbrev_str_count:
                return True
    return False
