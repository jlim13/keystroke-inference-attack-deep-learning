import sys
sys.path.append('../')

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_dataloders(syn_train_path, syn_test_path, real_train_path, real_test_path, args):
    
    pass
