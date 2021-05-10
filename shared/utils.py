import contextlib
import fasttext
import pickle
import re
import os

fasttext.FastText.eprint = print

def make_dirs(path):
    """ Create directory at given path 
    
    :param path: path name
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_pickle(filepath):
    """ Load data from pickle file from filepath name 
    
    :param filepath: filepath name
    """
    data = None
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    return data

def dump_pickle(data, filepath):
    """ Dump data to pickle format to a given filepath name
    
    :param filepath: filepath name
    """
    with open(filepath, "wb") as file:
        pickle.dump(data, file)
    
def read_txt_as_list(filepath):
    """ Load txt file content into list 
    
    :param filepath: filepath name
    :return: list of tokens
    """
    f = open(filepath, 'r+')
    data = [line.rstrip('\n') for line in f.readlines()]
    f.close()
    return data

def get_doc_length(text):
    """ Determine number of words in a document."""
    doc_length = len(re.findall(r'\w+', text))
    return doc_length

def load_lang_model(filepath):
    """ Load FastText model 
    :param filepath: 
    :return: FastText model
    """
    model = None
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model = fasttext.load_model(filepath)
    return model