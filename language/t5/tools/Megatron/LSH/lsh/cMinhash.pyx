# distutils: language = c++
# distutils: sources = lsh/MurmurHash3.cpp

__author__ = "Matti Lyra"

cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc
from libc.stdint cimport uint32_t, int32_t, uint64_t
import numpy as np
cimport numpy as np

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x64_128 (const void *key, int len, uint32_t seed, void *out) nogil

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x86_32 (const void *key, int len, uint32_t seed, void *out) nogil


@cython.boundscheck(False) # turn of bounds-checking for entire function
def minhash_64(char* c_str, int strlen,
               np.ndarray[dtype=np.uint32_t, ndim=1] seeds not None,
               int char_ngram):
    """Perform shingling and compute minhash of each shingle.

    Creates `char_ngram` length shingles from input string `c_str` and computes
    `len(seeds)` number 128bit min hashes for each shingle. A shingle is a
    character ngram of length `char_ngram`, consecutive shingles are taken over
    a sliding window.
    """
    cdef uint32_t num_seeds = len(seeds)
    cdef np.ndarray[np.uint64_t, ndim=1] fingerprint = \
        np.zeros((num_seeds, ), dtype=np.uint64)

    cdef uint64_t INT64_MAX = 9223372036854775807
    cdef uint64_t hashes[2]
    cdef uint64_t minhash

    # memory view to the numpy array - this should be free of any python
    cdef uint64_t [:] mem_view = fingerprint
    cdef uint32_t i, s
    with nogil:
        for s in range(num_seeds):
            minhash = INT64_MAX
            for i in range(strlen - char_ngram + 1):
                MurmurHash3_x64_128(c_str, char_ngram, seeds[s], hashes)
                if hashes[0] < minhash:
                    minhash = hashes[0]
                c_str += 1

            # store the current minhash
            mem_view[s] = minhash

            # reset string pointer for next hash
            c_str -= strlen - char_ngram + 1
    return fingerprint


@cython.boundscheck(False) # turn of bounds-checking for entire function
def minhash_32(char* c_str, int strlen,
               np.ndarray[dtype=np.uint32_t, ndim=1] seeds not None,
               int char_ngram):
    """Perform shingling and compute minhash of each shingle.

    Creates `char_ngram` length shingles from input string `c_str` and computes
    `len(seeds)` number 128bit min hashes for each shingle. A shingle is a
    character ngram of length `char_ngram`, consecutive shingles are taken over
    a sliding window.
    """
    cdef uint32_t num_seeds = len(seeds)
    cdef np.ndarray[np.uint32_t, ndim=1] fingerprint = \
        np.zeros((num_seeds, ), dtype=np.uint32)

    cdef int32_t INT32_MAX = 4294967295
    cdef int32_t hash_[1]
    cdef int32_t minhash

    # memory view to the numpy array - this should be free of any python
    cdef uint32_t [:] mem_view = fingerprint
    cdef uint32_t i, s
    with nogil:
        for s in range(num_seeds):
            minhash = INT32_MAX
            for i in range(strlen - char_ngram + 1):
                MurmurHash3_x86_32(c_str, char_ngram, seeds[s], hash_)
                if hash_[0] < minhash:
                    minhash = hash_[0]
                c_str += 1

            # store the current minhash
            mem_view[s] = minhash

            # reset string pointer for next hash
            c_str -= strlen - char_ngram + 1
    return fingerprint
