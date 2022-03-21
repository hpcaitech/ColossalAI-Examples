import os
import hashlib
import tarfile
import zipfile
import requests

DATASET = dict()
DATASET['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    '3c914d17d80b1459be871a5039ac23e752a53cbe'
)
DEFAULT_CACHE_PATH = 'data'

def download(name, cache_dir=os.path.join('..', DEFAULT_CACHE_PATH)):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATASET, f"{name} does not exist in {DATASET}."
    url, sha1_hash = DATASET[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir