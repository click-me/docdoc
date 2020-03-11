from nltk.stem import PorterStemmer
from sentence_splitter import split_text_into_sentences
import pickle
import pkg_resources


with open(pkg_resources.resource_filename(__name__, 'resource/stopwords.txt')) as f:
    stopwords = f.readlines()
    punctuations = ['.', ',', '/', '_', '-', '+', ';', ':', '(', ')', '[', ']', '*', '\'']
    ignorewords = stopwords + punctuations

with open(pkg_resources.resource_filename(__name__, 'resource/Terms_inventory.pkl'), 'rb') as f:
    inventory = pickle.load(f)
    #concepts = list(inventory.keys())

ps = PorterStemmer()


def find_all(str, substr):
    start = 0
    while True:
        start = str.find(substr, start)
        if start == -1: return
        yield start
        start += len(substr)  # use start += 1 to find overlapping matches


def split2sentences2tokens(text):
    sentences = [i for i in split_text_into_sentences(text=text, language='en', non_breaking_prefix_file=pkg_resources.resource_filename(__name__, 'resource/custom_english_non_breaking_prefixes.txt'))
                 if i.strip() != '']

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


def supported_concept_type():
    return list(inventory.keys())


def n_grams_match(text, concept_types):
    assert isinstance(concept_types, list), "Argument \'concept_types\' should be a list"
    assert set(concept_types).issubset(set(list(inventory.keys()))), "invalid concept types, please use method \'supported_concept_type()\' to check supported concept types"

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
