from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def generate_instances_for_NER(split_sentences_tokens, entities_spans):
    """ Generate training instances for NER tasks from (1)split_sentences_tokens and (2)entities_spans.
        For example:
            original_text = "Hello. It's me."
            split_sentences_tokens = [[(0, 5, 'Hello'), (5, 6, '.')],
                                      [(7, 9, 'It'), (9, 10, "'"), (10, 11, 's'), (12, 14, 'me'), (14, 15, '.')]]
            entities_spans = [(0, 5), (7, 14)]
            output = [(['Hello', '.'], ['B','O']),
                      (['It', "'", 's', 'me', '.'], ['B', 'I', 'I', 'I', 'O'])]
        Args:
            split_sentences_tokens: A list of tokens-list. Can be generated by class DocumentTokenizer's 'split2sentences2tokens' method.
            entities_spans: A list of entities spans.
        Returns:
            A list of sentences' tokens and corresponding BIO labels.
    """
    entities_spans = sorted(entities_spans, key=lambda i: (i[0], i[1]))

    generated_instances = []
    for sentence in split_sentences_tokens:
        sentence_tokens = [i[2] for i in sentence]
        sentence_labels = []
        for token in sentence:
            token_start_index = token[0]
            token_end_index = token[1]
            token_label = 'O'
            for span in entities_spans:
                entity_start_index = span[0]
                entity_end_index = span[1]
                if token_start_index < entity_start_index:
                    break
                if token_start_index == entity_start_index:
                    token_label = 'B'
                    break
                if entity_start_index < token_start_index and token_end_index <= entity_end_index:
                    token_label = 'I'
                    break
            sentence_labels.append(token_label)
        generated_instances.append((sentence_tokens, sentence_labels))
