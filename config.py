import argparse

def config():
    p = argparse.ArgumentParser("ResMultiHGNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataroot', type=str, default='/home/jing/data/HGNN', help='the directary of your .mat data')
    p.add_argument('--dataname', type=str, default='NTU2012', help='data name (ModelNet40/NTU2012)')
    p.add_argument('--model-name', type=str, default='HGNN', help='(HGNN, ResHGNN, MultiHGNN, ResMultiHGNN)')
    p.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')
    p.add_argument('--nhid', type=int, default=128, help='number of hidden features')
    p.add_argument('--dropout', type=float, default=0.5, help='dropout probability')

    p.add_argument('--epochs', type=int, default=600, help='number of epochs to train')
    p.add_argument('--patience', type=int, default=200, help='early stop after specific epochs')
    p.add_argument('--gpu', type=int, default=0, help='gpu number to use')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--nostdout', action="store_true",  help='do not output logging info to terminal')
    p.add_argument('--balanced', action="store_true",  help='only use the balanced subset of training labels')
    p.add_argument('--split-ratio', type=int, default=0,  help='if set unzero, this is for Task: Stability Analysis, new total/train ratio')
    p.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')
    return p.parse_args()