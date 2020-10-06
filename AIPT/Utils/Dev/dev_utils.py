import os
import time
import subprocess as sp
import importlib


def get_mod_time(filename, verbose=True):
    '''
    Finds last time file {filename} was modified. If {verbose}, prints UTC timestamp.
    
    @returns: Timestamp of last mod time, as a {time.struct_time}: https://docs.python.org/3/library/time.html#time.struct_time
    '''
    statbuf = os.stat(filename)
    mod_time = time.gmtime(statbuf.st_mtime)
    if verbose:
        print('File:', filename, '\n', "Last modified:", time.strftime("%Y-%m-%d %H:%M:%S", mod_time), 'UTC')
    return mod_time



def reload_aipt_module(module, aipt_path, verbose=False):
    '''
    reload an aipt module, useful when updating .py files in aipt
    '''
    python = sp.run(['which', 'python'], check=True, stdout=sp.PIPE, universal_newlines=True)
    out = sp.run(['python', 'setup.py', 'install'], check=True, stdout=sp.PIPE, universal_newlines=True,
                   cwd=aipt_path)
    if verbose:
        print(out.stdout)
    else:
        print(out.stdout.split('\n')[-1])
    importlib.import_module(module.__name__)
    importlib.reload(module)
    
    
def get_aipt_reload_fn(aipt_path):
    return lambda module, verbose=False: reload_aipt_module(module, aipt_path, verbose=verbose)