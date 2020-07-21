from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from itertools import groupby
from operator import itemgetter


def tokenIdx2charIdx(tokens_seq, labels_seq, group_by_sentence=True):
    """ convert token-level indices (model's immediate output) into character-level indices (for downstream task, such as display).
        For example:
            original_text = "Hello. It's me."
            tokens_seq = [[(0, 5, 'Hello'), (5, 6, '.')],
                          [(7, 9, 'It'), (9, 10, "'"), (10, 11, 's'), (12, 14, 'me'), (14, 15, '.')]]
            labels_seq = [['B', 'O'],
                          ['B', 'I', 'I', 'I', 'O']]
            output = [(0, 5), (7, 14)]
        Args:
            tokens_seq: A list of tokens-list. Can be generated by class DocumentTokenizer's 'split2sentences2tokens' method.
            labels_seq: A list of model's immediate output.
        Returns:
            A list of entities spans.
    """
    if group_by_sentence:
        # 1. Get token-level indices of target tokens
        tokenIdx_BI = [np.where(np.array(i) in ['B', 'I'])[0] for i in labels_seq]

        # 2. Get token-level indices of entities
        split_entity_tokenIdx = []
        for i in tokenIdx_BI:
            entity_tokenIdx = []
            for k, g in groupby(enumerate(i), lambda ix: ix[0] - ix[1]):
                entity_tokenIdx.append(list(map(itemgetter(1), g)))
            split_entity_tokenIdx.append(entity_tokenIdx)

        # 3. Get character-level indices of entities
        entity_charIdx = []
        for i in range(len(split_entity_tokenIdx)):
            if split_entity_tokenIdx[i] != []:
                tokens = tokens_seq[i]
                for var in split_entity_tokenIdx[i]:
                    begin_index = tokens[var[0]][0]
                    end_index = tokens[var[-1]][1]
                    entity_charIdx.append((begin_index, end_index))
        return entity_charIdx
    else:
        # 1. Get token-level indices of target tokens
        tokenIdx_BI = np.where(np.array(labels_seq) in ['B', 'I'])[0]

        # 2. Get token-level indices of entities
        entity_tokenIdx = []
        for k, g in groupby(enumerate(tokenIdx_BI), lambda ix: ix[0] - ix[1]):
            entity_tokenIdx.append(list(map(itemgetter(1), g)))

        # 3. Get character-level indices of entities
        entity_charIdx = []
        for var in entity_tokenIdx:
            begin_index = tokens_seq[var[0]][0]
            end_index = tokens_seq[var[-1]][1]
            entity_charIdx.append((begin_index, end_index))

        return entity_charIdx
