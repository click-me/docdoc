from sentence_splitter import split_text_into_sentences
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import pickle
import os


ps = PorterStemmer()
stopwords = set(stopwords.words('english'))
punctuations = ['.', ',', '/', '_', '-', '+', ';', ':', '(', ')', '[', ']', '*', '\'']
ignorewords = list(stopwords) + punctuations


def find_all(str, substr):
    start = 0
    while True:
        start = str.find(substr, start)
        if start == -1: return
        yield start
        start += len(substr)  # use start += 1 to find overlapping matches


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


def split(text):
    sentences = [i for i in split_text_into_sentences(text=text, language='en', non_breaking_prefix_file='resource/custom_english_non_breaking_prefixes.txt') if i != '']

    offsets = []
    offsets_separated_by_sentence = []

    start_index = 0

    for sentence in sentences:

        sentence = sentence.lower()
        sentence = separate_punctuation(sentence)

        words = sentence.split()
        text = text.lower()

        for word in words:
            word_index = text.index(word, start_index)
            word_len = len(word)
            start_index = word_index + word_len
            offsets.append((word, word_index, start_index))
            offsets_separated_by_sentence.append((word, word_index, start_index))

        offsets_separated_by_sentence.append(('***SENTENCE BREAKER***', -1, -1))

    return offsets, offsets_separated_by_sentence


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
        tokens.append((word, word_index, start_index))

    return tokens


# 
#
def stemming_match(text, stemmed_term_list):
    # 1. Raw <-> Stemmed Matching
    stemmed_text = ' '.join([ps.stem(j) for j in [i[0] for i in split2tokens(text)]])
    raw_index_list = split2tokens(text)
    stem_index_list = split2tokens(stemmed_text)
    ann_list = []
    for i in range(len(raw_index_list)):
        ann_list.append((i, stem_index_list[i][0], (stem_index_list[i][1], stem_index_list[i][2]), raw_index_list[i][0],
                         (raw_index_list[i][1], raw_index_list[i][2])))

    # 2. Get spans of concepts
    index_list = []
    for stemmed_term in stemmed_term_list:
        start_index_list = list(find_all(stemmed_text, ' ' + stemmed_term + " "))
        for i in start_index_list:
            index_list.append((i + 1, i + 1 + len(stemmed_term)))

    # 3. Remove duplicates and overlapping
    index_list = list(set(index_list))
    index_list = sorted(index_list)
    index_list = list(dict(index_list).items())
    new_index_list = []
    for i in range(len(index_list)):
        if (i == 0):
            new_index_list.append(index_list[i])
        else:
            if index_list[i][1] > index_list[i - 1][1]:
                new_index_list.append(index_list[i])
    index_list = new_index_list

    # 4. Get full annotations
    new_ann_list = []
    for ann in ann_list:
        token_index = ann[0]
        index = ann[2]

        label = 'O'
        for var in index_list:
            if index[0] < var[0]:
                break
            if index[0] == var[0]:
                label = 'B'
                break
            if index[0] > var[0] and index[1] <= var[1]:
                label = 'I'
                break
        new_ann_list.append((token_index, ann[1], ann[2], ann[3], ann[4], label))
    ann_list = new_ann_list

    return ann_list


def generate_instances_stemming(text, stemmed_terms):
    ann_list = stemming_match(text, stemmed_terms)
    label_list = [i[5] for i in ann_list]
    BI_token_index = [i[0] for i in ann_list if (i[-1] == 'B' or i[-1] == 'I')]

    sentences = [i for i in split_text_into_sentences(text=text, language='en', non_breaking_prefix_file='resource/custom_english_non_breaking_prefixes.txt') if i != '']
    split_token_index = []
    labels = []
    sentences_chosen_list = []

    curr_index = 0

    for i in range(len(sentences)):
        sentence = sentences[i]
        split_token_index.append((curr_index, curr_index + len(split2tokens(sentence))))
        labels.append(label_list[curr_index: curr_index + len(split2tokens(sentence))])
        for token_idx in BI_token_index:
            if curr_index <= token_idx < curr_index + len(split2tokens(sentence)):
                sentences_chosen_list.append(i)
                break
        curr_index = curr_index + len(split2tokens(sentence))

    # Generate training instances
    data_instances = []
    for i in sentences_chosen_list:
        sentence = sentences[i]
        label = labels[i]
        if label[0] == 'I':
            del data_instances[-1]
            sentence = sentences[i - 1] + ' ' + sentence
            label = labels[i - 1] + label
        data_instances.append(([i[0] for i in split2tokens(sentence)], label))

    return data_instances


def exact_match(text, terms):
    # 1. Tokenize 
    text = text.lower()
    token_list = split2tokens(text)
    ann_list = [(i, token_list[i][0], (token_list[i][1], token_list[i][2])) for i in range(len(token_list))]

    # 2. Get spans of concepts
    index_list = []
    for term in terms:
        start_index_list = list(find_all(text, " " + term + " "))
        for i in start_index_list:
            index_list.append((i + 1, i + 1 + len(term)))

    # 3. Remove duplicates and overlapping
    index_list = list(set(index_list))
    index_list = sorted(index_list)
    index_list = list(dict(index_list).items())
    new_index_list = []
    for i in range(len(index_list)):
        if i == 0:
            new_index_list.append(index_list[i])
        else:
            if index_list[i][1] > index_list[i - 1][1]:
                new_index_list.append(index_list[i])
    index_list = new_index_list

    # 4. Get full annotations
    new_ann_list = []
    for ann in ann_list:
        index = ann[2]

        label = 'O'
        for var in index_list:

            if index[0] < var[0]:
                break
            if index[0] == var[0]:
                label = 'B'
                break
            if index[0] > var[0] and index[1] <= var[1]:
                label = 'I'
                break
        new_ann_list.append((ann[0], ann[1], ann[2], label))
    ann_list = new_ann_list

    return ann_list


def generate_instances_exact(text, terms):
    ann_list = exact_match(text, terms)
    label_list = [i[3] for i in ann_list]
    BI_token_index = [i[0] for i in ann_list if (i[-1] == 'B' or i[-1] == 'I')]

    sentences = [i for i in split_text_into_sentences(text=text, language='en',
                                                      non_breaking_prefix_file='resource/custom_english_non_breaking_prefixes.txt')
                 if i != '']
    split_token_index = []
    labels = []
    sentences_chosen_list = []

    curr_index = 0

    for i in range(len(sentences)):
        sentence = sentences[i]
        split_token_index.append((curr_index, curr_index + len(split2tokens(sentence))))
        labels.append(label_list[curr_index: curr_index + len(split2tokens(sentence))])
        for token_idx in BI_token_index:
            if curr_index <= token_idx and token_idx < curr_index + len(split2tokens(sentence)):
                sentences_chosen_list.append(i)
                break
        curr_index = curr_index + len(split2tokens(sentence))

    # Generate training instances
    data_instances = []
    for i in sentences_chosen_list:
        sentence = sentences[i]
        label = labels[i]
        if label[0] == 'I':
            del data_instances[-1]
            sentence = sentences[i - 1] + ' ' + sentence
            label = labels[i - 1] + label
        data_instances.append(([i[0] for i in split2tokens(sentence)], label))

    return data_instances


def build_concept_inventory(concept_type, snomed_path, word2remove_pickle):
    ps = PorterStemmer()

    df = pd.read_csv(snomed_path)
    with open(word2remove_pickle, "rb") as f:
        ambiguous_terms = pickle.load(f)

    df = df[(df.type == concept_type) & (~df['term'].isin(ambiguous_terms))]

    df = pd.DataFrame(data={'conceptId': df['conceptId'],
                            'term': df['term'].map(lambda x: separate_punctuation(x).lower())})
    df['stem_term'] = df['term'].map(lambda x: ' '.join([ps.stem(i) for i in x.split()]))
    df['compare'] = [' '.join(sorted([n for n in i.split() if n not in ignorewords])) for i in
                     df['stem_term'].to_list()]
    df.to_csv("Terms/SNOMED_%s.csv" % concept_type.replace(' ', '_'), index=False)


def generate_data_instances_for_concept(corpus_df, concept_type, method, update_rate):
    df = pd.read_csv("SNOMED_%s.csv" % concept_type.replace(' ', '_'))
    terms = df['term'].to_list()
    stemmed_terms = [i for i in df['stem_term'].to_list() if i not in stopwords]

    # 1. Read cache
    instances_list = [i for i in os.listdir() if
                      i.startswith('instances_%s_%s_' % (concept_type.replace(' ', '_'), method))]
    numbers = [int(i.split('.')[0][12 + len(concept_type) + len(method):]) for i in instances_list]

    if numbers != []:
        numbers.sort()
        number = numbers[-1]
        with open("instances_%s_%s_%s.pkl" % (concept_type.replace(' ', '_'), method, str(number)), "rb") as f:
            instances = pickle.load(f)
    else:
        number = 0
        instances = []

    # 2. Generate new
    for index, row in corpus_df.iterrows():
        row_id = row['ROW_ID']
        if row_id <= number:
            continue
        text = row['TEXT']

        if method == 'exact':
            instances = instances + generate_instances_exact(text, terms)
        elif method == 'stemming':
            instances = instances + generate_instances_stemming(text, stemmed_terms)
        else:
            break

        if int(row_id) % update_rate == 0:
            with open("instances_%s_%s_%s.pkl" % (concept_type.replace(' ', '_'), method, str(row_id)), 'wb') as f:
                pickle.dump(instances, f)
            print(row_id)


def n_grams_match(text, inventory, concept_types):
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
            segment_list += [([n[0] for n in i], i[0][1], i[-1][2]) for i in grams
                             if (i[0][0] not in ignorewords
                                 and (i[-1][0] not in ignorewords)
                                 and len(set([j[0] for j in i]).intersection(set(punctuations))) == 0)]

        for i in segment_list:
            if ' '.join(sorted([ps.stem(n) for n in i[0] if n not in ignorewords])) in terms['compare']:
                tags.append(i)
        output[concept_type] = tags
    return output


def n_grams_match_sentence(text, inventory, concept_types):
    # 0. Format text
    text = remove_fake_line_breaker(text)
    
    # 1. Match entities
    ents_by_type = n_grams_match(text, inventory, concept_types)

    # 2. Sort entities
    ents = []
    for concept_type in ents_by_type.keys():
        ents += [{"start": i[1], "end": i[2], "label": concept_type} for i in ents_by_type[concept_type] if i != []]
    ents = sorted(ents, key=lambda i: i['start'])

    # 3. Split sentences
    sentences = [i for i in split_text_into_sentences(text=text, language='en',
                                                      non_breaking_prefix_file='resource/custom_english_non_breaking_prefixes.txt')
                 if i != '']

    # 4. Index sentence 
    sentences_index = []
    start_index = 0

    for sen in sentences:
        sen_index = text.index(sen.split()[0], start_index)
        sen_len = len(sen)
        start_index = sen_index + sen_len
        sentences_index.append((sen_index, start_index))

    # 5. Group entities
    ents_by_sen = [[] for i in range(len(sentences_index))]
    sen_index = 0
    for ent in ents:
        while ent['end'] > sentences_index[sen_index][1]:
            sen_index += 1
        if sentences_index[sen_index][0] < ent['start'] and ent['end'] < sentences_index[sen_index][1]:
            ents_by_sen[sen_index].append(ent)

    # 6. Reindex entities
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
