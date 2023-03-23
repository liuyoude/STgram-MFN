import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import STgramMFN
from trainer import Trainer
from dataset import ASDDataset
import utils

sep = os.sep

def main(args):
    # set random seed
    utils.setup_seed(args.random_seed)
    # set device
    cuda = args.cuda
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True
    # load data
    # train_dirs = args.train_dirs
    train_dirs = args.train_dirs + args.add_dirs
    args.meta2label, args.label2meta = utils.metadata_to_label(train_dirs)
    train_file_list = []
    for train_dir in train_dirs:
        train_file_list.extend(utils.get_filename_list(train_dir))
    train_dataset = ASDDataset(args, train_file_list, load_in_memory=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    # set model
    args.num_classes = len(args.meta2label.keys())
    args.logger.info(f'Num classes: {args.num_classes}')
    net = STgramMFN(num_classes=args.num_classes, use_arcface=args.use_arcface,
                    m=float(args.m), s=float(args.s), sub=args.sub_center)
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)
    # optimizer & scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.1*float(args.lr))
    # trainer
    trainer = Trainer(args=args,
                      net=net,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      transform=train_dataset.transform)
    # train model
    if not args.load_epoch:
        trainer.train(train_dataloader)
    # test model
    load_epoch = args.load_epoch if args.load_epoch else 'best'
    model_path = os.path.join(args.writer.log_dir, 'model', f'{load_epoch}_checkpoint.pth.tar')
    if args.dp:
        trainer.net.module.load_state_dict(torch.load(model_path)['model'])
    else:
        trainer.net.load_state_dict(torch.load(model_path)['model'])
    trainer.test(save=True, gmm_n=args.gmm_n)
    trainer.evaluator(save=True, gmm_n=args.gmm_n)


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    args = parser.parse_args()
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'STgram-MFN(m={args.m},s={args.s})'
    args.version = f'{time_str}-{args.version}' if not args.load_epoch and args.time_version else args.version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # save version files
    if args.save_version_files: utils.save_load_version_files(log_dir, args.save_version_file_patterns, args.pass_dirs)
    # run
    args.writer, args.logger = writer, logger
    args.logger.info(args.version)
    main(args)
    # save config file
    utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))


if __name__ == '__main__':
    run()