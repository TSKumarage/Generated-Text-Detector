import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import argparse
from tqdm import tqdm

import random
import time

from Data import TrainDataset
from Modules import DCVAE


class GeneratedTextDetection:

    def __init__(self, args):
        torch.manual_seed(int(time.time()))

        self.args = args

        self.dataset = TrainDataset(mbsize=self.args.mb_size, custom_data=True, eval=True,
                                    train_data_path=self.args.eval_data,
                                    eval_data_file=self.args.eval_data,
                                    checkpoint_path=self.args.vocab_file)

        self.model = self._init_detector()

    def _init_detector(self):

        model = DCVAE(
            self.dataset.n_vocab, self.args.h_dim, self.args.z_dim,
            self.args.c_dim, p_word_dropout=0.3, freeze_embeddings=True,
            gpu=self.args.gpu
        )

        if self.args.gpu:
            model.load_state_dict(torch.load((self.args.check_point + '{}.bin').format(self.args.model_name)))
        else:
            model.load_state_dict(torch.load((self.args.check_point + '{}.bin').format(self.args.model_name),
                                                  map_location=lambda storage, loc: storage))

        return model

    def evaluate(self):
        results = {"score": [], "evidence": {"coherency_score": [], "comsense_consistency": []}}

        with torch.no_grad():
            for data in self.dataset.test_loader:
                if self.args.gpu:
                    inputs = data.text.cuda()
                else:
                    inputs = data.text

                disc_out = self.model.forward_discriminator(inputs.transpose(0, 1))

                _, predicted = torch.max(disc_out, 1)

                results["score"].extend(predicted.tolist())

                results["evidence"]["coherency_score"].extend([random.uniform(0, 1) for i in range(len(predicted))])

                results["evidence"]["comsense_consistency"].extend(
                    [random.uniform(0, 1) for i in range(len(predicted))])

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Conditional Generated Text Detection: Discriminator'
    )

    # Input data and files
    parser.add_argument('--model_name', default="dcvae", type=str,
                        help='name of the model')
    parser.add_argument('--eval_data', default="/content/test.csv", type=str,
                        help='input data file for evaluation')
    parser.add_argument('--vocab_file', default="/content/vocab.pkl", type=str,
                        help='saved vocab')
    parser.add_argument('--check_point', default="/content/", type=str,
                        help='saved model checkpoint directory')

    # Model parameters
    parser.add_argument('--mb_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--z_dim', default=64, type=int,
                        help='latent code size')
    parser.add_argument('--h_dim', default=64, type=int,
                        help='hiden dim size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--lr_decay_every', default=1000000, type=int,
                        help='learning decay')
    parser.add_argument('--n_iter', default=20000, type=int,
                        help='number of iterations')
    parser.add_argument('--log_interval', default=1000, type=int,
                        help='log interval')
    parser.add_argument('--c_dim', default=2, type=int,
                        help='condition var size')

    parser.add_argument('--gpu', default=False, type=bool,
                        help='whether to run in the GPU')
    parser.add_argument('--save', default=False, action='store_true',
                        help='whether to save model or not')

    args = parser.parse_args()

    artifact = GeneratedTextDetection(args)
    results = artifact.evaluate()


if __name__ == "__main__":
    main()


