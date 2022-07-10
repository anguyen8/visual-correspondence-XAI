r""" Functions to download semantic correspondence datasets """

import tarfile
import os

import requests

from . import pfpascal
from . import pfwillow
from . import spair


def load_dataset(benchmark, datapath, thres, split='test'):
    r""" Instantiate a correspondence dataset """
    correspondence_benchmark = {
        'spair': spair.SPairDataset,
        'pfpascal': pfpascal.PFPascalDataset,
        'pfwillow': pfwillow.PFWillowDataset
    }

    dataset = correspondence_benchmark.get(benchmark)
    if dataset is None:
        raise Exception('Invalid benchmark dataset %s.' % benchmark)

    return dataset(benchmark, datapath, thres, split)


def download_from_google(token_id, filename):
    r""" Download desired filename from Google drive """

    print('Downloading %s ...' % os.path.basename(filename))

    url = 'https://docs.google.com/uc?export=download'
    destination = filename + '.tar.gz'
    session = requests.Session()

    response = session.get(url, params={'id': token_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': token_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)
    file = tarfile.open(destination, 'r:gz')

    print("Extracting %s ..." % destination)
    file.extractall(filename)
    file.close()

    os.remove(destination)
    os.rename(filename, filename + '_tmp')
    os.rename(os.path.join(filename + '_tmp', os.path.basename(filename)), filename)
    os.rmdir(filename+'_tmp')


def get_confirm_token(response):
    r"""Retrieves confirm token"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    r"""Saves the response to the destination"""
    chunk_size = 32768

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)


def download_dataset(datapath, benchmark):
    r"""Downloads semantic correspondence benchmark dataset from Google drive"""
    if not os.path.isdir(datapath):
        os.mkdir(datapath)

    file_data = {        
        # 'spair': ('1s73NVEFPro260H1tXxCh1ain7oApR8of', 'SPair-71k') old version
        'spair': ('1KSvB0k2zXA06ojWNvFjBv0Ake426Y76k', 'SPair-71k'),
        'pfpascal': ('1OOwpGzJnTsFXYh-YffMQ9XKM_Kl_zdzg', 'PF-PASCAL'),
        'pfwillow': ('1tDP0y8RO5s45L-vqnortRaieiWENQco_', 'PF-WILLOW')
    }

    file_id, filename = file_data[benchmark]
    abs_filepath = os.path.join(datapath, filename)

    if not os.path.isdir(abs_filepath):
        download_from_google(file_id, abs_filepath)
