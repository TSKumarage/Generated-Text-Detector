import json
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from download import download


def load_texts(data_file, label=False, expected_size=None):
    texts = []

    for line in tqdm(open(data_file), desc=f'Loading {data_file}'):
        texts.append(json.loads(line)['text'])

    if label:
        label = []
        for line in tqdm(open(data_file), desc=f'Loading {data_file}'):
            label.append(json.loads(line)['label'])

        return texts, label

    return texts


class Corpus:
    def __init__(self, name, data_dir='data', label=False, skip_train=False, single_file=False):
        # download(name, data_dir=data_dir)
        self.name = name

        if single_file:

            if label:
                self.data, self.label = load_texts(f'{data_dir}/{name}.jsonl', label=True)
            else:
                self.data = load_texts(f'{data_dir}/{name}.jsonl')

        else:

            self.train = load_texts(f'{data_dir}/{name}.train.jsonl') if not skip_train else None
            self.test = load_texts(f'{data_dir}/{name}.test.jsonl')
            self.valid = load_texts(f'{data_dir}/{name}.valid.jsonl')


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.real_texts, self.fake_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            if index < len(self.real_texts):
                text = self.real_texts[index]
                label = 0
            else:
                text = self.fake_texts[index - len(self.real_texts)]
                label = 1

        tokens = self.tokenizer.encode(text, max_length=self.max_sequence_length, truncation=True)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.max_len - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask, label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label


class EncodedSingleDataset(Dataset):
    def __init__(self, input_texts: List[str], input_labels:List[int], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.input_texts = input_texts
        self.input_labels = input_labels
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return self.epoch_size or len(self.input_texts)

    def __getitem__(self, index):

        text = self.input_texts[index]
        label = self.input_labels[index]

        tokens = self.tokenizer.encode(text, max_length=self.max_sequence_length, truncation=True)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.max_len - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()


        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask, label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label


class EncodeEvalData(Dataset):
    def __init__(self, input_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return self.epoch_size or len(self.input_texts)

    def __getitem__(self, index):
        text = self.input_texts[index]

        tokens = self.tokenizer.encode(text, max_length=self.max_sequence_length, truncation=True)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.max_len - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))

        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask