from sentence_splitter import split_text_into_sentences
import pkg_resources
from spacy.lang.en import English
from bert import tokenization

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


class DocumentTokenizer(object):
    """Runs end-to-end tokenization."""

    def __init__(self, sentence_splitter='spacy_sentence_splitter', tokenizer='bert_basic_tokenizer',
                 do_lower_case=True):
        self.sentence_splitter_fn = sentence_splitter_dict.get(sentence_splitter, spacy_sentence_splitter)
        self.tokenizer = tokenizer_class_dict.get(tokenizer, tokenization.BasicTokenizer)(do_lower_case)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        sentences = self.sentence_splitter_fn(text=text, do_lower_case=self.do_lower_case)

        if self.do_lower_case:
            text = text.lower()

        output_sentences = []
        next_sentence_start_index = 0

        for sen in sentences:
            next_token_start_index = next_sentence_start_index
            temp = []
            tokens = self.tokenizer.tokenize(sen)
            for tok in tokens:
                token_index = text.index(tok, next_token_start_index)
                next_token_start_index = token_index + len(tok)
                temp.append((token_index, next_token_start_index, tok))
            output_sentences.append(temp)
            next_sentence_start_index = temp[-1][1]

        return output_sentences


def default_sentence_splitter(text, do_lower_case=True):
    split_sentences = split_text_into_sentences(text=text, language='en',
                                                non_breaking_prefix_file=pkg_resources.resource_filename(__name__,
                                                                                                         'resource/custom_english_non_breaking_prefixes.txt'))
    if do_lower_case:
        return [i.lower() for i in split_sentences if i.strip() != '']
    else:
        return [i for i in split_sentences if i.strip() != '']


def spacy_sentence_splitter(text, do_lower_case=True):
    split_sentences = []
    lines = text.split('\n')
    for line in lines:
        if line.strip() != '':
            split_sentences.extend([sent.string.strip() for sent in nlp(line).sents])
    if do_lower_case:
        return [i.lower() for i in split_sentences if i.strip() != '']
    else:
        return [i for i in split_sentences if i.strip() != '']


sentence_splitter_dict = {"default_sentence_splitter": default_sentence_splitter,
                          "spacy_sentence_splitter": spacy_sentence_splitter}

tokenizer_class_dict = {"bert_basic_tokenizer": tokenization.BasicTokenizer}
