import numpy as np
import pytest

from lsh.cache import Cache
from lsh.minhash import MinHasher


@pytest.fixture
def default_hasher():
    return MinHasher(seeds=100)


@pytest.fixture
def default_cache(default_hasher):
    return Cache(default_hasher)


def is_nondecreasing(L):
    # http://stackoverflow.com/a/4983359/419338
    return all(x <= y for x, y in zip(L, L[1:]))


@pytest.mark.parametrize("char_ngram", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("hashbytes", [4, 8])
@pytest.mark.parametrize("num_bands", [20, 40, 50])
@pytest.mark.parametrize("seed", range(3))
def test_cache(char_ngram, hashbytes, num_bands, seed):
    hasher = MinHasher(seeds=200, char_ngram=char_ngram,
                       hashbytes=hashbytes, random_state=seed)
    lsh = Cache(hasher, num_bands=num_bands)
    # very small band width => always find duplicates

    short_doc = 'This is a simple document'
    another_doc = 'Some text about animals.'
    long_doc = 'A much longer document that contains lots of information\
       different words. The document produces many more shingles.'

    assert not lsh.is_duplicate(short_doc)
    lsh.add_doc(short_doc, 0)
    assert lsh.get_duplicates_of(short_doc) == {0}
    assert lsh.is_duplicate(short_doc, doc_id=0)
    assert lsh.is_duplicate(short_doc)

    assert not lsh.is_duplicate(long_doc)
    lsh.add_doc(long_doc, 1)
    lsh.add_doc(another_doc, 2)
    assert lsh.is_duplicate(another_doc)

    assert lsh.is_duplicate(long_doc, doc_id=1)
    assert lsh.is_duplicate(long_doc)

    words = long_doc.split()
    long_doc_missing_word = ' '.join([words[0]] + words[2:])

    assert lsh.get_duplicates_of(long_doc_missing_word) == {1}
    assert lsh.is_duplicate(long_doc_missing_word)
    assert lsh.is_duplicate(long_doc + ' Word.')

    assert lsh.get_all_duplicates() == set()
    lsh.add_doc(long_doc_missing_word, 3)
    assert lsh.get_all_duplicates() == {(1, 3)}

    lsh.add_doc(long_doc_missing_word, 4)
    assert lsh.get_all_duplicates() == {(1, 3), (1, 4), (3, 4)}


mc_long_doc = "Jang MC Min Chul is a Protoss player from South Korea, who " \
              "last played for Trig  Esports before retiring. On May 23rd, " \
              "2016, MC announced his return to pro-gaming by joining CJ " \
              "Entus. He is currently "

mc_med_doc = "Jang MC Min Chul is a Protoss player from South Korea, who " \
             "last played for Trig Esports before retiring. He is currently "

mc_short_doc = "Jang MC Min Chul is currently "


@pytest.mark.parametrize("doc", [mc_long_doc, mc_med_doc, mc_short_doc])
def test_num_bands(doc):
    """
    add near-duplicate documents to three caches with different settings
    check that hashers with low band_width finds more matches (over 50 runs)
    """
    suffixes = ['teamless', 'retired', 'awesome', 'overweight']
    duplicates = []
    divisors_of_200 = [4, 10, 20, 25, 40, 50, 100]

    for seed in range(10):
        hasher = MinHasher(seeds=200, char_ngram=5, random_state=seed)
        caches = [Cache(hasher, num_bands=n) for n in divisors_of_200]

        for c in caches:
            c.add_doc(doc + suffixes[0], 0)

        for s in suffixes[1:]:
            duplicates.append([c.is_duplicate(doc + s) for c in caches])

    sums = np.array(duplicates).sum(axis=0)
    print(sums)
    assert is_nondecreasing(sums)


@pytest.mark.parametrize("doc", [mc_long_doc, mc_med_doc, mc_short_doc])
def test_real_world_usage(default_cache, doc):
    default_cache.add_doc(doc, 0)
    default_cache.add_doc(doc, 1)

    assert default_cache.is_duplicate(doc)
    assert default_cache.is_duplicate(doc, 0)
    assert default_cache.is_duplicate(doc, 1)
    assert default_cache.is_duplicate(doc, 2)


def test_filtering_by_jaccard(default_cache):
    data = {0: mc_long_doc, 1: mc_med_doc,
            2: mc_med_doc, 3: mc_short_doc}

    for id, doc in data.items():
        default_cache.add_doc(doc, id)

    for mj in np.arange(0.1, 0.91, step=0.1):
        dupes = default_cache.get_all_duplicates(min_jaccard=mj)
        assert dupes == {(1, 2)}

    dupes = default_cache.get_duplicates_of(doc=mc_med_doc,
                                            min_jaccard=0.9)
    assert dupes == {1, 2}

    dupes = default_cache.get_duplicates_of(doc_id=1,
                                            min_jaccard=0.9)
    assert dupes == {1, 2}

    dupes = default_cache.get_duplicates_of('Nothing to see',
                                            min_jaccard=0.1)
    assert dupes == set()


def test_jaccard(default_hasher):
    assert default_hasher.jaccard("This is a doc", "This is a doc") == 1

    high_j = default_hasher.jaccard("This is a doc", "That is a doc")
    low_j = default_hasher.jaccard("This is a doc", "Cats in a tree")
    assert 0 <= low_j < high_j <= 1


@pytest.mark.parametrize("num_bands", [3, 6, 7, 9, 71, 99, 101])
def test_invalid_settings(num_bands, default_hasher, default_cache):
    with pytest.raises(AssertionError):
        lsh = Cache(default_hasher, num_bands=num_bands)
        lsh.add_doc('Hi', 1)
        lsh.get_duplicates_of('Hello')

    default_cache.add_doc('Hi', 0)
    with pytest.raises(ValueError):
        default_cache.get_duplicates_of(doc_id=123)


def test_clear(default_cache):
    default_cache.add_doc(mc_long_doc, 0)
    assert default_cache.is_duplicate(mc_long_doc)
    f = default_cache.hasher.fingerprint(mc_long_doc)

    default_cache.clear()
    f1 = default_cache.hasher.fingerprint(mc_long_doc)

    assert not default_cache.is_duplicate(mc_long_doc)
    np.testing.assert_array_equal(f, f1)


def test_remove_by_id(default_cache):
    default_cache.add_doc(mc_long_doc, 0)
    default_cache.add_doc(mc_med_doc, 1)
    default_cache.add_doc(mc_short_doc, 2)
    default_cache.add_doc(mc_short_doc, 3)

    # initially everything is a duplicate
    assert default_cache.is_duplicate(mc_long_doc)
    assert default_cache.is_duplicate(mc_med_doc)
    assert default_cache.is_duplicate(mc_short_doc)

    # doc removed, it must no longer be a dupe, but all others still are
    default_cache.remove_id(0)
    assert not default_cache.is_duplicate(mc_long_doc)
    assert default_cache.is_duplicate(mc_med_doc)
    assert default_cache.is_duplicate(mc_short_doc)

    # another doc removed. non-removed docs are still duplicates
    default_cache.remove_id(1)
    assert not default_cache.is_duplicate(mc_long_doc)
    assert not default_cache.is_duplicate(mc_med_doc)
    assert default_cache.is_duplicate(mc_short_doc)

    default_cache.remove_id(2)
    assert not default_cache.is_duplicate(mc_long_doc)
    assert not default_cache.is_duplicate(mc_med_doc)
    assert default_cache.is_duplicate(mc_short_doc)

    default_cache.remove_id(3)
    assert not default_cache.is_duplicate(mc_short_doc)

    with pytest.raises(KeyError):
        default_cache.remove_id(123)  # unknown id


def test_remove_by_text(default_cache):
    default_cache.add_doc(mc_long_doc, 0)
    default_cache.add_doc(mc_short_doc, 1)
    default_cache.add_doc(mc_short_doc, 2)

    assert default_cache.is_duplicate(mc_long_doc)
    assert default_cache.is_duplicate(mc_short_doc)

    # both occurences of the removed doc should go away
    default_cache.remove_doc(mc_short_doc)
    assert default_cache.is_duplicate(mc_long_doc)
    assert not default_cache.is_duplicate(mc_short_doc)
