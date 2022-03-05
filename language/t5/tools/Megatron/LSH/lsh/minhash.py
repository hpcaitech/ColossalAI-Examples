# -*- coding: utf-8 -*-
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

import numpy as np

from .cMinhash import minhash_32, minhash_64


class MinHasher(object):
    def __init__(self, seeds, char_ngram=8, random_state=None, hashbytes=8):
        """The MinHasher creates fingerprints from raw documents.

        The MinHasher facilitates the creation of MinHash document
        fingerprints. It creates overlapping character ngram shingles of length
        `char_ngram` using a sliding window over the document. To preprocessing
        to the documents is done, they are shingled as is.

        Parameters:
        -----------
        seeds: np.ndarray, int
            A Numpy array of 32bit unsigned integers to use as seeds to
            initialise hash functions, or a single integer for the number of
            seeds to create. A minhash is computed for each hash function
            derived from seeds.

        char_ngram: int
            The number of consecutive characters to include in a sliding window
            when creating the document shingles.

        random_state: None, int, np.random.RandomState
            A random state to initialise the random number generator with.
        """
        self.char_ngram = char_ngram
        random_state = np.random.RandomState(random_state)
        if hashbytes not in set([4, 8, 16]):
            raise ValueError('Hash has to be 4, 8 or 16 bytes.')

        if hashbytes == 16:
            raise NotImplementedError()

        self.hashbytes = hashbytes
        if isinstance(seeds, np.ndarray):
            self._seeds = seeds.astype(np.uint32)
        else:
            self._seeds = np.array(random_state.randint(0, 1e6, seeds),
                                   dtype=np.uint32)

    @property
    def num_seeds(self):
        return len(self._seeds)

    @lru_cache(maxsize=10000)
    def fingerprint(self, text):
        if isinstance(text, str):
            text = text.encode('utf8')
        if self.hashbytes == 4:
            fingerprint = minhash_32(text, len(text),
                                     self._seeds, self.char_ngram)
        elif self.hashbytes == 8:
            fingerprint = minhash_64(text, len(text),
                                     self._seeds, self.char_ngram)
        return fingerprint

    def jaccard(self, doc1, doc2):
        if isinstance(doc1, str):
            f_a = set(self.fingerprint(doc1))
        else:
            f_a = doc1  # assume it's z fingerprint
        if isinstance(doc1, str):
            f_b = set(self.fingerprint(doc2))
        else:
            f_b = doc2
        return len(f_a & f_b) / len(f_a | f_b)
