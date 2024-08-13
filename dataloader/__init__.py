from datasets import dataset_factory
from .bert import *
from .sas import *
from .bert_finetune import *
from .bert_finetune import *


def dataloader_factory(args, model_code, oracle_model=None, distill=False):
    dataset = dataset_factory(args)

    if model_code == 'bert':
        dataloader = BERTDataloader(args, dataset, pretrained_model=oracle_model, distill=distill)
    elif model_code == 'sas':
        dataloader = SASDataloader(args, dataset, pretrained_model=oracle_model, distill=distill)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
