import torch
import torch.nn.functional as F


import path, os 

from utils.utils import *
from models import *


import config 
args = config.config()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

seed = args.seed
torch.manual_seed(seed)




# initialize parameters
dataroot = path.Path(args.dataroot)
datadir = dataroot / f'{args.dataname}_mvcnn_gvcnn.mat'
model_name = args.model_name
k_nearest = 10

Xs, y, mask_train, mask_val  = load_data(datadir, selected_mod=(0, 1))

if args.balanced:
    mask_train, mask_val = get_balanced_sub_idx(y, mask_train)

split_ratio = args.split_ratio
if split_ratio:
    mask_train, mask_val = get_split(y, 1. / split_ratio)



# init H and X
tmpDataDir = path.Path(f'data/{args.dataname}')
tmpDataDir.makedirs_p()

Hs = [
    generate_H(Xs[imod], k_nearest, tmpDataDir / f'transductive_{imod}_k={k_nearest}.pt')
    for imod in (0, 1)
]

if 'Multi' in model_name:
    Hs = [ create_sparse_H(H).cuda() for H in Hs  ] 
    Xs = [ X.cuda() for X in Xs]
    nfeats = [X.shape[1] for X in Xs]
    
    Xtrs = [X.clone() for X in Xs]
    for X in Xtrs: X[mask_val] = 0 
    
else:
    Hs = create_sparse_H(hyedge_concat(Hs)).cuda()
    Xs = torch.hstack(Xs).cuda()
    nfeats = Xs.shape[1]
    
    Xtrs = Xs.clone()
    Xtrs[mask_val] = 0

y, mask_train, mask_val = y.cuda(), mask_train.cuda(), mask_val.cuda()





nclass = y.max().item() + 1
nlayer, nhid, dropout = args.nlayer, args.nhid, args.dropout

Models = {
    'HGNN': HGNN,
    'ResHGNN': ResHGNN,
    'MultiHGNN': MultiHGNN,
    'ResMultiHGNN': ResMultiHGNN,
}

model = Models[model_name](args, nfeats, nhid, nclass, nlayer, dropout).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.9)




#### config the output directory
dataname = args.dataname
out_dir = path.Path( f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}_{split_ratio}/{seed}' )


import shutil 
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()


from logger import get_logger
baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
baselogger.info(args)


def train():
    model.train()
    optimizer.zero_grad()
    if args.dataname == 'ModelNet40':
        pred = model(Xtrs, Hs)
    else:
        pred = model(Xs, Hs)
    loss = F.nll_loss(pred[mask_train], y[mask_train])
    loss.backward()
    optimizer.step()
    if schedular: schedular.step()
    return loss 


def val():
    model.eval()
    pred = model(Xs, Hs)

    _train_acc = accuracy(pred[mask_train], y[mask_train])
    _val_acc = accuracy(pred[mask_val], y[mask_val])

    return _train_acc, _val_acc



best_acc, badcounter = 0.0, 0
for epoch in range(1, args.epochs+1):
    loss = train()
    train_acc, val_acc = val()
    if val_acc > best_acc:
        best_acc = val_acc
        badcounter = 0 
    else:
        badcounter += 1
    if badcounter > args.patience: break
    baselogger.info(f'Epoch: {epoch}, Loss: {loss:.4f}, Train:{train_acc:.2f}, Val:{val_acc:.2f}, Best Val acc:{best_acc:.3f}' )
