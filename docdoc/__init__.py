from nltk.stem import PorterStemmer
from sentence_splitter import split_text_into_sentences
import pickle
import pkg_resources
import numpy as np
from itertools import groupby
from operator import itemgetter


with open(pkg_resources.resource_filename(__name__, 'resource/stopwords.txt')) as f:
    STOPWORDS = f.readlines()
    PUNCTUATIONS = ['.', ',', '/', '_', '-', '+', ';', ':', '(', ')', '[', ']', '*', '\'']
    IGNOREWORDS = STOPWORDS + PUNCTUATIONS

with open(pkg_resources.resource_filename(__name__, 'resource/terms_inventory.pkl'), 'rb') as f:
    INVENTORY = pickle.load(f)

ps = PorterStemmer()


def find_all(str, substr):
    start = 0
    while True:
        start = str.find(substr, start)
        if start == -1: return
        yield start
        start += len(substr)  # use start += 1 to find overlapping matches


def split2sentences2tokens(text):

    sentences = [separate_punctuation(i.lower()) for i in split_text_into_sentences(text=text, language='en', non_breaking_prefix_file=pkg_resources.resource_filename(__name__, 'resource/custom_english_non_breaking_prefixes.txt'))
                 if i.strip() != '']
    text = text.lower()

    output_sentences = []
    next_sentence_start_index = 0

    for sen in sentences:
        next_token_start_index = next_sentence_start_index
        temp = []
        tokens = sen.strip().split()
        for tok in tokens:
            token_index = text.index(tok, next_token_start_index)
            next_token_start_index = token_index + len(tok)
            temp.append((token_index, next_token_start_index, tok))
        output_sentences.append(temp)
        next_sentence_start_index = temp[-1][1]

    return output_sentences

def split2sentences(text):
    output_sentences = split2sentences2tokens(text)
    out_raw_sentences = [(var[0][0], var[-1][1], text[var[0][0]:var[-1][1]]) for var in output_sentences]
    return out_raw_sentences


def split2tokens(text):
    output_sentences = split2sentences2tokens(text)
    output_tokens = sum(output_sentences, [])
    return output_tokens


def tokenIdx2charIdx(token_seq, label_seq):
    # 0-->B
    # 1-->I
    # 2-->O

    assert len(token_seq) == len(label_seq), "The length of \'token_seq\' is not equal to that of \'label_seq\'"
    assert token_seq != [] and (type(token_seq[0]) is tuple) and len(token_seq[0]) == 3, "Invalid argumant \'token_seq\', please use \'docdoc.split2tokens\' to generate a valid \'token_seq\' variable"

    # 1. Get indices of target tokens
    label_seq = np.array(label_seq)
    tokenIdx_BI = np.where(label_seq != 2)[0]

    # 2. Get token-level indices of entities
    entity_tokenIdx = []
    for k, g in groupby(enumerate(tokenIdx_BI), lambda ix: ix[0] - ix[1]):
        entity_tokenIdx.append(list(map(itemgetter(1), g)))

    # 3. Get character-level indices of entities
    entity_charIdx = []
    for var in entity_tokenIdx:
        begin_index = token_seq[var[0]][0]
        end_index = token_seq[var[-1]][1]
        entity_charIdx.append((begin_index, end_index))

    return entity_charIdx


def separate_punctuation(sentence):
    for var in PUNCTUATIONS:
        sentence = sentence.replace(var, ' %s '%var)
    return sentence


def supported_concept_type():
    return list(INVENTORY.keys())



def remove_fake_line_breaker(text):
    fake_breakers = []
    breakers = [i for i in find_all(text, '\n')]
    for i in breakers:
        if i == breakers[-1] or (i + 1) in breakers:
            continue
        if (not text[i + 1].isupper()) or text[i - 1] == ',':
            fake_breakers.append(i)
    for i in fake_breakers:
        text = text[:i] + ' ' + text[i + 1:]
    return text


def process_term(term):
    term = term.lower()
    term = separate_punctuation(term)
    term = ' '.join(sorted([ps.stem(i) for i in term.split() if i not in IGNOREWORDS]))
    return term


def n_grams_match(text, terms, N):
    # 1. Tokenize
    text = remove_fake_line_breaker(text)
    tokens = split2tokens(text)

    # 2. Matching
    selected_segments = []

    # 2.1 1-grams
    for tok in tokens:
        if tok[2] in terms:
            selected_segments.append(tok)

    # 2.2 n-grams
    processed_terms = [process_term(i) for i in terms]
    for n in range(2, N + 1):
        n_grams = [tokens[i:i + n] for i in range(len(tokens) - n + 1)]
        for var in n_grams:
            start_index = var[0][0]
            end_index = var[-1][1]
            tok_list = [i[2] for i in var]
            if (tok_list[0] not in IGNOREWORDS and (tok_list[-1] not in IGNOREWORDS) and len(
                    set([i[2] for i in var]).intersection(set(PUNCTUATIONS))) == 0):
                if process_term(' '.join(tok_list)) in processed_terms:
                    selected_segments.append((start_index, end_index, ' '.join(tok_list)))

    sorted_segments = sorted(selected_segments, key=lambda i: (i[0], i[1]))

    # 3. Group entities by sentences
    sentences_spans = [(i[0], i[1]) for i in split2sentences(text)]
    segments_by_sentence = [[] for i in range(len(sentences_spans))]
    sen_index = 0
    for seg in sorted_segments:
        seg_startIdx = seg[0]
        seg_endIdx = seg[1]
        while seg_endIdx > sentences_spans[sen_index][1]:
            sen_index += 1
        if sentences_spans[sen_index][0] < seg_startIdx and seg_endIdx < sentences_spans[sen_index][1]:
            segments_by_sentence[sen_index].append(seg)

    # 4. Reindex entities
    entities_by_sentence = []
    for i in range(len(segments_by_sentence)):
        sen_startIdx = sentences_spans[i][0]
        sen_endIdx = sentences_spans[i][1]
        sentence = text[sen_startIdx: sen_endIdx]
        segment_spans = [i[0:2] for i in segments_by_sentence[i]]
        entity_spans = cleanup_spanList(segment_spans)
        entity_spans_reindex = [(i[0] - sen_startIdx, i[1] - sen_startIdx) for i in entity_spans]
        entities_by_sentence.append({'text': sentence, 'entities': entity_spans_reindex})

    return entities_by_sentence

def cleanup_spanList(spanList):
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
