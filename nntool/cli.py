"""
nntool

Usage:
    nntool create_project <name>
    nntool train --topology <name> --dataset <name> --epochs <number> --batchsize <number> [--regularization]
    nntool -h | --help
    nntool --version

Options:
    -h --help       Show this screen
    --version       Show version

Examples:
    nntool create_project project1
    nntool train --topology top.txt --dataset iris --epochs 200 --batchsize 120 --regularization
"""


from docopt import docopt
from nntool.trainer import Trainer
from nntool.datasets import *
import os


from . import __version__ as VERSION

def main():
    """Main CLI entrypoint."""
    options = docopt(__doc__, version=VERSION)
    if options['create_project'] is True and options['<name>'] is not None:
        pr_name = options['<name>'][0]
        os.makedirs(pr_name+'/weights')
        os.makedirs(pr_name+'/scripts')
        os.makedirs(pr_name+'/csvresults')
        os.makedirs(pr_name+'/datasets')
    elif options['train'] and options['--topology'] and options['--dataset'] and options['<name>'] is not None:
        trainer = Trainer(options['<name>'][0])
        data_name=options['<name>'][1]
        dsets = Datasets()
        if data_name == 'iris':
            train_examples, train_labels, test_examples, test_labels, set_names = dsets.iris()
        elif data_name == 'mnist':
            train_examples, train_labels, test_examples, test_labels, set_names = dsets.mnist()
        elif data_name == 'cfar10':
            train_examples, train_labels, test_examples, test_labels, set_names = dsets.cifar10()
        elif data_name == 'chars74k':
            train_examples, train_labels, test_examples, test_labels, set_names = dsets.chars74k_num_caps_fonts()
        elif data_name == 'fashion_mnist':
            train_examples, train_labels, test_examples, test_labels, set_names = dsets.fashion_mnist()
        elif data_name == 'mnist-chars74k':
            train_examples, train_labels, test_examples, test_labels, set_names = MnistChars74k().get_dataset()


        else:
            assert 1==0, 'not find dataset'
        trainer.SetData(train_examples, train_labels, test_examples, test_labels, set_names)
        trainer.SetSession()
        n_epoch = int(options['<number>'][0])
        batch_size = int(options['<number>'][1])
        regularization=bool(options['--regularization'])
        trainer.TRAIN(n_epoch, batch_size, Tb=3, Te=1, test_predict=True, regularization=regularization)
        trainer.save_weights()