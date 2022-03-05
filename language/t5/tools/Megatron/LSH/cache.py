# -*- coding: utf-8 -*-
from __future__ import division

import json
from collections import defaultdict
import itertools
import logging
from copy import deepcopy

import numpy as np

__author__ = "Matti Lyra"


class Cache(object):
    """LSH provides a way of determining the local neighbourhood of a document.

    Locality Sensitive Hashing relies on probabilistic guarantees of a hash
    function family to produce hash collisions for similar content. The
    implementation uses MinHash to produce those collisions and allows for fast
    deduplication of data sets without having to do all pairs comparisons.
    """

    def __init__(self, hasher, num_bands=10, **kwargs):
        # each fingerprint is divided into n bins (bands) and duplicate
        # documents are computed only for documents that land in the same
        # bucket in one of the bins
        # bins[idx of band where docs may overlap][hash of fingerprint] ->
        # list of doc ids that have that fingerprint segment at that position
        self.bins = [defaultdict(set) for _ in range(num_bands)]
        self.hasher = hasher
        msg = 'The number of seeds in the fingerprint must ' \
              'be divisible by the number of bands'
        assert hasher.num_seeds % num_bands == 0, msg
        self.band_width = hasher.num_seeds // num_bands
        self.num_bands = num_bands

        self.fingerprints = dict()

    def bins_(self, fingerprint):
        yield from enumerate(np.array_split(fingerprint, self.num_bands))

    def clear(self):
        self.bins = [defaultdict(set) for _ in range(self.num_bands)]
        self.hasher.fingerprint.cache_clear()

    def add_doc(self, doc, doc_id):
        fingerprint = self.hasher.fingerprint(doc.encode('utf8'))
        self.add_fingerprint(fingerprint, doc_id)

    def add_fingerprint(self, fingerprint, doc_id):
        self.fingerprints[doc_id] = fingerprint
        for bin_i, bucket in self.bins_(fingerprint):
            # todo faster hash here? or no hash at all?
            bucket_id = hash(tuple(bucket))
            self.bins[bin_i][bucket_id].add(doc_id)

    def filter_candidates(self, candidate_id_pairs, min_jaccard):
        logging.info('Computing Jaccard sim of %d pairs',
                     len(candidate_id_pairs))
        res = set()
        for id1, id2 in candidate_id_pairs:
            # todo id1, id2 may not be contained in data
            jaccard = self.hasher.jaccard(self.fingerprints[id1],
                                          self.fingerprints[id2])
            if jaccard > min_jaccard:
                res.add((id1, id2))
        logging.info('Keeping %d/%d candidate duplicate pairs',
                     len(res), len(candidate_id_pairs))
        return res

    def remove_id(self, doc_id):
        fingerprint = self.fingerprints[doc_id]
        for bin_i, bucket in self.bins_(fingerprint):
            bucket_id = hash(tuple(bucket))
            self.bins[bin_i][bucket_id].remove(doc_id)

        del self.fingerprints[doc_id]

    def remove_doc(self, doc):
        fingerprint = self.hasher.fingerprint(doc.encode('utf8'))
        doc_ids = {id for id, finger in self.fingerprints.items()
                  if all(a == b for a, b in zip(finger, fingerprint))}
        for i in doc_ids:
            self.remove_id(i)

    def get_all_duplicates(self, min_jaccard=None):
        candidate_pairs = set()
        for b in self.bins:
            for bucket_id in b:
                if len(b[bucket_id]) > 1:
                    pairs_ = set(itertools.combinations(b[bucket_id], r=2))
                    candidate_pairs.update(pairs_)
        if min_jaccard is None:
            return candidate_pairs

        return self.filter_candidates(candidate_pairs, min_jaccard)

    def get_duplicates_of(self, doc=None, doc_id=None, min_jaccard=None):
        if doc_id is not None and doc_id in self.fingerprints:
            fingerprint = self.fingerprints[doc_id]
        elif doc is not None:
            fingerprint = self.hasher.fingerprint(doc.encode('utf8'))
        else:
            raise ValueError('Must provide a document or a known document id')

        candidates = set()
        for bin_i, bucket in self.bins_(fingerprint):
            bucket_id = hash(tuple(bucket))
            candidates.update(self.bins[bin_i][bucket_id])

        if min_jaccard is None:
            return candidates
        else:
            return {x for x in candidates
                    if self.hasher.jaccard(fingerprint,
                                           self.fingerprints[x]) > min_jaccard}

    def is_duplicate(self, doc, doc_id=None):
        return len(self.get_duplicates_of(doc, doc_id=doc_id)) > 0
