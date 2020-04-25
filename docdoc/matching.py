from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.stem import PorterStemmer
import pkg_resources
from docdoc import find_all

with open(pkg_resources.resource_filename(__name__, 'resource/stopwords.txt')) as f:
    STOPWORDS = [i.strip() for i in f.readlines()]
    PUNCTUATIONS = ['.', ',', '/', '_', '-', '+', ';', ':', '(', ')', '[', ']', '*', '\'']
    IGNOREWORDS = STOPWORDS + PUNCTUATIONS

ps = PorterStemmer()


def n_grams_match(text, terms, tokenizer, N):
    # 1. Tokenize
    #text = remove_fake_line_breaker(text)
    tokens = tokenizer.split2tokens(text)

    # 2. Matching
    selected_segments = []

    # 2.1 1-grams
    for tok in tokens:
        if tok[2] in terms:
            selected_segments.append(tok)

    # 2.2 n-grams
    processed_terms = [process_term(i, tokenizer) for i in terms]
    for n in range(2, N + 1):
        n_grams = [tokens[i:i + n] for i in range(len(tokens) - n + 1)]
        for var in n_grams:
            start_index = var[0][0]
            end_index = var[-1][1]
            tok_list = [i[2] for i in var]
            if (tok_list[0] not in IGNOREWORDS and (tok_list[-1] not in IGNOREWORDS) and len(
                    set([i[2] for i in var]).intersection(set(PUNCTUATIONS))) == 0):
                if process_term(' '.join(tok_list), tokenizer) in processed_terms:
                    selected_segments.append((start_index, end_index, ' '.join(tok_list)))

    sorted_segments = sorted(selected_segments, key=lambda i: (i[0], i[1]))

    # 3. filter by sentences
    sentences_spans = [(i[0], i[1]) for i in tokenizer.split2sentences(text)]
    filtered_segments = []
    for i in sorted_segments:
        start_index = i[0]
        end_index = i[1]
        for sentence_span in sentences_spans:
            if (sentence_span[0] <= start_index) and (end_index <= sentence_span[1]):
                filtered_segments.append(i)
                break

    return filtered_segments


def process_term(term, tokenizer):
    term = term.lower()
    try:
        subterms = [i[2] for i in tokenizer.split2tokens(term)]
        process_term = ' '.join(sorted([ps.stem(i) for i in subterms if i not in IGNOREWORDS]))
        return process_term
    except:
        return term


def remove_fake_line_breaker(text):
    def is_fake_line_breaker(text, i):
        if text[i + 1].islower():
            return True
        if text[i - 1] in [',']:
            return True
        return False

    fake_breakers = []
    text = text.strip()
    breakers = [i for i in find_all(text, '\n')]
    for i in breakers:
        if is_fake_line_breaker(text, i):
            fake_breakers.append(i)
    for i in fake_breakers:
        text = text[:i] + ' ' + text[i + 1:]
    return text


def clean_up_span_list(spanList):
    spanList = list(set(spanList))
    spanList = sorted(spanList)
    spanList = list(dict(spanList).items())

    output = []
    for i in range(len(spanList)):
        if i == 0:
            output.append(spanList[i])
        else:
            if spanList[i][1] > spanList[i - 1][1]:
                output.append(spanList[i])
    return output
