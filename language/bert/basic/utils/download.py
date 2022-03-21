import os
import tarfile
import zipfile
import requests

DATASET = dict()
DATASET['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
)
DATASET['wikitext-103'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
)
DEFAULT_CACHE_PATH = 'data'

def download(name, cache_dir=os.path.join('..', DEFAULT_CACHE_PATH)):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATASET, f"{name} does not exist in {DATASET}."
    url = DATASET[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        print(f'Cached {fname} is loading...')
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