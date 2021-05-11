import contextlib
import fasttext
import pickle
import json
import re
import os

fasttext.FastText.eprint = print

def make_dirs(path):
    """ Create directory at given path 
    
    :param path: path name
    """
    if not os.path.exists(path):
        os.makedirs(path)

def dump_json(data, filepath, indent=4, sort_keys=True):
    """ Dump dictionary to json file to a given filepath name 
    
    :param data: python dictionary
    :param filepath: filepath name
    :param indent: indent keys in json file
    :param sort_keys: boolean flag to sort keys
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)

def load_json(filepath):
    """ Load json file from filepath into a dictionary 
    
    :param filepath:
    :return: python dictionary
    """
    data = dict()
    with open(filepath) as data_file:    
        data = json.load(data_file)
    return data

def dump_txt(data, filepath):
    """ Dump data to txt file format to a given filepath name 
    
    :param filepath: filepath name
    """
    with open(filepath, "w") as file :
        file.write(data)

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