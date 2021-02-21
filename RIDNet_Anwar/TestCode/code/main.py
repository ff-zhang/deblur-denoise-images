import torch

from utility import checkpoint
from data import Data
from model import Model
from loss import Loss
from option import args
from trainer import Trainer
from torch.nn import DataParallel

torch.manual_seed(args.seed)
checkpoint = checkpoint(args)

print("\n" + str(args))

if checkpoint.ok:
    loader = Data(args)

    model = Model(args, checkpoint)
    model = DataParallel(model)
    loss = Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

