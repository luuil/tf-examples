from __future__ import print_function

from urllib.request import urlretrieve
from urllib.error import HTTPError
from os.path import exists
from os import makedirs

def check_exists(path):
  return exists(path)

def check_path(path):
  if not check_exists(path):
    makedirs(path)

def download_file(url, path):
  try:
    if not check_exists(fpath):
      print('downloading [{url}] -> [{path}] ..'.format(url=url, path=path))
      urlretrieve(url, path)
    else:
      print('{path} exists, skipping..'.format(path=path))
  except HTTPError as e:
    print(url, e.msg)

if __name__ == "__main__":

  # Default: current directory.
  # You can specify your own location
  fdir = '.'

  url = 'http://yann.lecun.com/exdb/mnist'
  to_be_download = ['train-images-idx3-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz']
  check_path(fdir)
  for filename in to_be_download:
    furl = '{url}/{name}'.format(url=url, name=filename)
    fpath = '{dir}/{name}'.format(dir=fdir, name=filename)
    download_file(furl, fpath)
