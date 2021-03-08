import torch
import argparse
from tqdm import tqdm

import random
import time

from torch.utils.data import DataLoader
from dataset import Corpus, EncodedDataset, EncodeEvalData

from transformers import *
from detector import RobertaForTextGenClassification

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

import decimal

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)


def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer,
                  max_sequence_length, random_sequence_length):

    real_corpus = Corpus(real_dataset, data_dir=data_dir, single_file=True)

    if fake_dataset == "TWO":
        real_train, real_valid = real_corpus.train * 2, real_corpus.valid * 2
        fake_corpora = [Corpus(name, data_dir=data_dir) for name in ['grover_fake', 'gpt2_fake']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])

    else:
        fake_corpus = Corpus(fake_dataset, data_dir=data_dir, single_file=True)

        real_valid = real_corpus.data
        fake_valid = fake_corpus.data

    min_sequence_length = 10 if random_sequence_length else None

    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer, max_sequence_length, min_sequence_length)
    validation_loader = DataLoader(validation_dataset)

    return validation_loader


def direct_load_dataset(data_dir, dataset, tokenizer,
                  max_sequence_length, random_sequence_length=False):

    data_corpus = Corpus(dataset, data_dir=data_dir, single_file=True)

    data_list = data_corpus.data

    validation_dataset = EncodeEvalData(data_list, tokenizer, max_sequence_length)

    validation_loader = DataLoader(validation_dataset)

    return validation_loader


class GeneratedTextDetection:
    """
    Artifact class
    """

    def __init__(self, args):
        torch.manual_seed(int(time.time()))

        self.args = args

        # Load the model from checkpoints
        self.model, self.tokenizer = self._init_detector()

    def _init_detector(self):

        model_name = 'roberta-large' if self.args.large else 'roberta-base'
        tokenization_utils.logger.setLevel('ERROR')
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForTextGenClassification.from_pretrained(model_name).to(self.args.device)

        # Load the model from checkpoints
        if self.args.device == "cpu":
            model.load_state_dict(torch.load((self.args.check_point + '{}.pt').format(self.args.model_name),
                                             map_location='cpu')['model_state_dict'])
        else:
            model.load_state_dict(
                torch.load((self.args.check_point + '{}.pt').format(self.args.model_name))['model_state_dict'])

        return model, tokenizer

    def evaluate(self, input_text):
        """
           Method that runs the evaluation and generate scores and evidence
        """

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        total = 0

        predict_prob = []
        y = []

        validation_loader = load_datasets(self.args.data_dir,
                                          self.args.real_dataset,
                                          self.args.fake_dataset,
                                          self.tokenizer,
                                          self.args.max_sequence_length,
                                          random_sequence_length=False)

        self.model.eval()

        with tqdm(validation_loader, desc="Eval") as loop:
            for texts, masks, labels in loop:

                total += labels.size(0)
                texts, masks, labels = texts.to(self.args.device), masks.to(self.args.device), labels.to(self.args.device)

                output_dic = self.model(texts, attention_mask=masks)
                disc_out = output_dic["logits"]

                _, predicted = torch.max(disc_out, 1)

                prob_values = [item[1] for item in disc_out.tolist()]

                predict_prob.extend(prob_values)

                y.extend(labels.tolist())

                # predict_prob.extend(predicted.tolist())

                tp += ((predicted == labels) & (labels == 1)).sum().item()
                tn += ((predicted == labels) & (labels == 0)).sum().item()
                fn += ((predicted != labels) & (labels == 1)).sum().item()
                fp += ((predicted != labels) & (labels == 0)).sum().item()

        recall = float(tp) / (tp+fn)
        precision = float(tp) / (tp+fp)
        f1_score = 2 * float(precision) * recall / (precision + recall)

        print('Accuracy of the discriminator: %d %%' % (
            100 * (tp+tn) / total))
        print('Recall of the discriminator: %d %%' % (
            100 * recall))
        print('Precision of the discriminator: %d %%' % (
            100 * precision))
        print('f1_score of the discriminator: %d %%' % (
            100 * f1_score))

        # calculate scores
        lr_auc = roc_auc_score(y, predict_prob)

        # summarize scores
        print('Classifier: ROC AUC=%.3f' % (lr_auc))

        # calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(y, predict_prob)

        eq_fpr = list(float_range(0, 1, 1 / len(lr_fpr)))
        eq_tpr = [1 - item for item in eq_fpr]

        # plot the roc curve for the model
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='RobertaTextGen')
        pyplot.plot(eq_fpr, eq_tpr, marker='.', label='EER')
        # axis labels

        pyplot.xlabel('Probability of False Alarm')
        pyplot.ylabel('Probability of Detection')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
        pyplot.savefig('ROC.pdf')


def main():
    parser = argparse.ArgumentParser(
        description='Roberta: Discriminator'
    )

    # Input data and files
    parser.add_argument('--model_name', default="drfinetunedroberta", type=str,
                        help='name of the model')
    parser.add_argument('--check_point', default="/content/drive/Shareddrives/DARPA/Datasets/Eval1Sources/", type=str,
                        help='saved model checkpoint directory')
    parser.add_argument('--data-dir', type=str, default='/content/drive/Shareddrives/DARPA/Datasets/Eval1Sources')
    parser.add_argument('--real-dataset', type=str, default='dryrun_real_eval.valid')
    parser.add_argument('--fake-dataset', type=str, default='dryrun_fake_eval.valid')

    # Model parameters
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')

    args = parser.parse_args()

    if args.device is None:
        args.device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'

    artifact = GeneratedTextDetection(args)

    sample_eval_data = ["Russian opposition politician Aleksei Navalny says he owes his life to the pilots who made an emergency landing when he collapsed on a flight last month, and to paramedics he said had quickly diagnosed poisoning and injected him with atropine.",
                        "To receive Steve Gutterman's Week In Russia each week via e-mail, subscribe by clicking here.Alyaksandr Lukashenka's rushed, hushed-up inauguration ceremony in Belarus may evoke memories for Vladimir Putin, and contain a warning about the future as Russia hurtles toward 2024. Also, a COVID-19 surge, Kremlin contortions over the poisoning of Aleksei Navalny, and a UN speech \"rehearsing the defense of a nation in decline.",
                        "As Americans anxiously watch the spread of coronavirus variants that were first identified in Britain and South Africa, scientists are finding a number of new variants that seem to have originated in the United States — and many of them may pose the same kind of extra-contagious threat."]

    artifact.evaluate(sample_eval_data)


if __name__ == "__main__":
    main()


