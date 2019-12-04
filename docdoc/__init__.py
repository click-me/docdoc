from sentence_splitter import split_text_into_sentences
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import pickle
import os

ps = PorterStemmer()
stopwords = stopwords.words('english')
punctuations = ['.', ',', '/', '_', '-', '+', ';', ':', '(', ')', '[', ']', '*', '\'']
ignorewords = stopwords + punctuations

with open('docdoc/resource/Terms_inventory.pkl', 'rb') as f:
    inventory = pickle.load(f)
    concepts = list(inventory.keys())


def find_all(str, substr):
    start = 0
    while True:
        start = str.find(substr, start)
        if start == -1: return
        yield start
        start += len(substr)  # use start += 1 to find overlapping matches


def split2sentences(text):
    sentences = [i for i in split_text_into_sentences(text=text, language='en',
                                                      non_breaking_prefix_file='docdoc/resource/custom_english_non_breaking_prefixes.txt')
                 if i != '']

    sentences_index = []
    start_index = 0

    for sen in sentences:
        sen_index = text.index(sen.split()[0], start_index)
        sen_len = len(sen)
        start_index = sen_index + sen_len
        sentences_index.append((sen_index, start_index, text[sen_index: start_index]))

    return sentences_index


def split2tokens(text):
    tokens = []
    start_index = 0

    sentence = text.lower()
    sentence = separate_punctuation(sentence)

    words = sentence.split()
    text = text.lower()

    for word in words:
        word_index = text.index(word, start_index)
        word_len = len(word)
        start_index = word_index + word_len
        tokens.append((word_index, start_index, word))

    return tokens




def separate_punctuation(sentence):
    sentence = sentence.replace('.', ' . ')
    sentence = sentence.replace(',', ' , ')
    sentence = sentence.replace('/', ' / ')
    sentence = sentence.replace('_', ' _ ')
    sentence = sentence.replace('-', ' - ')
    sentence = sentence.replace('+', ' + ')
    sentence = sentence.replace(';', ' ; ')
    sentence = sentence.replace(':', ' : ')
    sentence = sentence.replace('(', ' ( ')
    sentence = sentence.replace(')', ' ) ')
    sentence = sentence.replace('[', ' [ ')
    sentence = sentence.replace(']', ' ] ')
    sentence = sentence.replace('*', ' * ')
    sentence = sentence.replace('\'', ' \' ')

    return sentence


def split2sentences2tokens(text):
    sentences = split2sentences(text)

    text = text.lower()
    tokensInText = []
    start_index = 0

    for sent in sentences:

        tokensInSentence = []

        sentence = sent[2].lower()
        sentence = separate_punctuation(sentence)

        tokens = sentence.split()
        for token in sentence.split():
            token_start = text.index(token, start_index)
            token_end = token_start + len(token)
            tokensInSentence.append((token_start, token_end, token))
            start_index = token_end

        tokensInText.append(tokensInSentence)

    return tokensInText


def supported_concept_type():
    return concepts


def n_grams_match(text, concept_types):
    
    assert isinstance(concept_types, list), "Argument \'concept_types\' should be a list"
    assert set(concept_types).issubset(set(concepts)), "invalid concept types, please use method \'supported_concept_type()\' to check supported concept types"
    
    # 1. Tokenize
    text = text.lower()
    token_list = split2tokens(text)

    output = {}

    for concept_type in concept_types:
        tags = []
        terms = inventory[concept_type]

        # 2.1 1-grams
        for i in token_list:
            if i[0] in terms['term']:
                tags.append(i)

        # 2.2 n-grams
        segment_list = []
        for N in range(2, 6):
            grams = [token_list[i:i + N] for i in range(len(token_list) - N + 1)]
            segment_list += [([n[2] for n in i], i[0][0], i[-1][1]) for i in grams
                             if (i[0][2] not in ignorewords
                                 and (i[-1][2] not in ignorewords)
                                 and len(set([j[2] for j in i]).intersection(set(punctuations))) == 0)]

        for i in segment_list:
            if ' '.join(sorted([ps.stem(n) for n in i[0] if n not in ignorewords])) in terms['compare']:
                tags.append(i)
        output[concept_type] = tags
    return output


def n_grams_match_by_sentence(text, concept_types):
    # 0. Format text
    text = remove_fake_line_breaker(text)

    # 1. Match entities
    ents_by_type = n_grams_match(text, concept_types)

    # 2. Sort entities
    ents = []
    for concept_type in ents_by_type.keys():
        ents += [{"start": i[1], "end": i[2], "label": concept_type} for i in ents_by_type[concept_type] if i != []]
    ents = sorted(ents, key=lambda i: i['start'])

    # 3. Split sentences
    sentences_index = [(i[0], i[1]) for i in split2sentences(text)]

    # 4. Group entities
    ents_by_sen = [[] for i in range(len(sentences_index))]
    sen_index = 0
    for ent in ents:
        while ent['end'] > sentences_index[sen_index][1]:
            sen_index += 1
        if sentences_index[sen_index][0] < ent['start'] and ent['end'] < sentences_index[sen_index][1]:
            ents_by_sen[sen_index].append(ent)

    # 5. Reindex entities
    ents_per_sen = []

    for i in range(len(ents_by_sen)):
        start_index = sentences_index[i][0]
        sentence = text[sentences_index[i][0]: sentences_index[i][1]]

        ents = ents_by_sen[i]
        if ents != []:
            ents_reindex = [
                {'start': ent['start'] - start_index, 'end': ent['end'] - start_index, 'label': ent['label']} for ent in
                ents]
        else:
            ents_reindex = []

        ents_per_sen.append({'text': sentence, 'ents': ents_reindex})
    return ents_per_sen


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