import os.path

from .base import AbstractDataloader

import torch
import random
import torch.utils.data as data_utils
import numpy as np
import torch.nn.functional as F


class BERTFinetuneDataloader():
    def __init__(self, args, dataset, pretrained_model=None, load_finetune_seqs=False):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        # user and item IDs start from 1
        # self.orig_train = dataset['train']
        # self.orig_val = dataset['val']
        # self.orig_test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        if load_finetune_seqs and os.path.isfile('./sequence pattern/finetune seqs %s %f %d.npy' % (args.dataset_code, args.finetune_ratio, args.bottom_m)):
            all_seqs = np.load('./sequence pattern/finetune seqs %s %f %d.npy' % (args.dataset_code, args.finetune_ratio, args.bottom_m)).tolist()
            num_seqs = len(all_seqs)
        else:
            self.max_len = args.bert_max_len
            self.CLOZE_MASK_TOKEN = self.item_count + 1
            num_seqs = int(args.finetune_ratio * self.user_count)
            batch_size = 50
            num_batches = int(num_seqs / batch_size)
            num_seqs = num_batches * batch_size
            pretrained_model.eval()
            all_seqs = []
            for i in range(num_batches):
                mask_items = torch.tensor([self.CLOZE_MASK_TOKEN] * batch_size).to(args.device)
                # seqs = torch.ones(num_seqs, 1) * self.target_item
                # seqs = seqs.to(args.device)
                seqs = torch.randint(1, self.item_count + 1, (batch_size, 1)).to(args.device)
                for j in range(self.max_len - 1):
                    input_seqs = torch.zeros((batch_size, self.max_len)).to(args.device)
                    input_seqs[:, (self.max_len - 2 - j):-1] = seqs
                    input_seqs[:, -1] = mask_items
                    labels = pretrained_model(input_seqs.long())[:, -1, :]

                    _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                    sorted_items = sorted_items[:, :100] + 1

                    randomized_label = torch.rand(sorted_items.shape).to(args.device)
                    randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                    randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)

                    selected_indices = torch.distributions.Categorical(
                        F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                    row_indices = torch.arange(sorted_items.size(0))
                    seqs = torch.cat((seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)
                seqs = seqs.cpu().detach().tolist()
                all_seqs += seqs
            if not os.path.isdir('./sequence pattern'):
                os.mkdir('./sequence pattern')
            np.save('./sequence pattern/finetune seqs %s %f %d.npy' % (args.dataset_code, args.finetune_ratio, args.bottom_m), all_seqs)
        self.train, self.val, self.test = {}, {}, {}
        for i in range(num_seqs):
            new_user_idx = i + 1
            single_seq = all_seqs[i]
            self.train[new_user_idx] = single_seq[:-2]
            self.val[new_user_idx] = [single_seq[-2]]
            self.test[new_user_idx] = [single_seq[-1]]

        self.user_count = len(self.train)

        self.valid_users = sorted(self.train.keys())
        self.test_users = self.valid_users

        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        complete_item_set = set(self.smap.values())
        self.val_negative_samples, self.test_negative_samples = {}, {}
        for user in self.train.keys():
            # seen = set(self.train[user])
            # seen.update(self.val[user])
            # seen.update(self.test[user])
            # self.val_negative_samples[user] = list(complete_item_set-seen)
            # self.test_negative_samples[user] = list(complete_item_set-seen)
            self.val_negative_samples[user] = []
            self.test_negative_samples[user] = []
        # self.seen_samples = list(seen)

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BERTTrainDataset(
            self.train, self.max_len, self.mask_prob, self.max_predictions, self.sliding_size,
            self.CLOZE_MASK_TOKEN, self.item_count, self.rng, self.user_count)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = BERTValidDataset(self.train, self.val, self.max_len, self.CLOZE_MASK_TOKEN, self.val_negative_samples, self.user_count, valid_users=self.valid_users)
        elif mode == 'test':
            dataset = BERTTestDataset(self.train, self.val, self.test, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples, test_users=self.test_users)
        return dataset


class BERTTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, max_predictions, sliding_size, mask_token, num_items, rng, user_count):
        self.user_count = user_count  # number of benign users
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.max_predictions = max_predictions
        self.sliding_step = int(sliding_size * max_len)
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

        assert self.sliding_step > 0
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]
            if u <= self.user_count:
                self.benign_seqs_count = len(self.all_seqs)

    def __len__(self):
        return len(self.all_seqs)
        # return len(self.users)

    def __getitem__(self, index):
        # user = self.users[index]
        # seq = self._getseq(user)
        seq = self.all_seqs[index]

        tokens = []
        labels = []
        covered_items = set()
        for i in range(len(seq)):
            s = seq[i]
            if (len(covered_items) >= self.max_predictions) or (s in covered_items):
                tokens.append(s)
                labels.append(0)
                continue

            temp_mask_prob = self.mask_prob
            if i == (len(seq) - 1):
                temp_mask_prob += 0.1 * (1 - self.mask_prob)

            prob = self.rng.random()
            if prob < temp_mask_prob:
                covered_items.add(s)
                prob /= temp_mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BERTValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, user_count, valid_users=None):
        self.user_count = user_count
        self.u2seq = u2seq  # train
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class BERTTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, mask_token, negative_samples, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]  # append validation item after train seq
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)