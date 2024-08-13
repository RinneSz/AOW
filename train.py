from datasets import DATASETS
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from utils import *


def train(args, export_root=None, resume=False):
    args.lr = 0.001

    oracle_model = None
    if not args.gold:
        if args.dataset_code == 'ml-1m':
            args.num_items = 3416
        elif args.dataset_code == 'ml-20m':
            args.num_items = 18345
        elif args.dataset_code == 'steam':
            args.num_items = 13046
        elif args.dataset_code == 'beauty':
            args.num_items = 54542
        else:
            raise NotImplementedError('Please specify number of items!')
        if args.model_code == 'bert':
            oracle_model = BERT(args)
        elif args.model_code == 'sas':
            oracle_model = SASRec(args)
        else:
            raise NotImplementedError('Model not recognized!')

        root = 'experiments/' + args.model_code + '/' + args.dataset_code
        try:
            oracle_model.load_state_dict(
                torch.load(os.path.join(root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
        except:
            raise ValueError('Please train the oracle with --gold first!')
        oracle_model = oracle_model.to(args.device)

    if args.gold:
        fix_random_seed_as(args.model_init_seed)
    # fix_random_seed_as(args.model_init_seed)

    train_loader, val_loader, test_loader = dataloader_factory(args, args.model_code, oracle_model=oracle_model)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)

    if args.gold:
        export_root = 'experiments/' + args.model_code + '/' + args.dataset_code
    else:
        export_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + args.model_code + '/' + \
                      args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                      '_' + str(args.pattern_len) + '_' + str(args.bottom_m)
    
    if resume:
        try: 
            model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')

    # For ML-1M, train from scratch. For other datasets, fine-tune them with the pretrained oracle model to speed up the training process.
    if not args.gold and args.dataset_code != 'ml-1m':
        gold_model_root = 'experiments/' + args.model_code + '/' + args.dataset_code
        model.load_state_dict(torch.load(os.path.join(gold_model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))

    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    if args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.train()
    trainer.test(test_watermark=False)


if __name__ == "__main__":
    set_template(args)

    batch = 128
    if args.gold or args.dataset_code == 'ml-1m':
        args.num_epochs = 1000
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    train(args, resume=False)
