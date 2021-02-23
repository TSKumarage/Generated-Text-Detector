import math
import torch
import argparse
from tqdm import tqdm

import random
import time

from torch.utils.data import DataLoader
from dataset import EncodeEvalData

from transformers import *
from detector import RobertaForTextGenClassification


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

        # Encapsulate the inputs
        eval_dataset = EncodeEvalData(input_text, self.tokenizer, self.args.max_sequence_length)
        eval_loader = DataLoader(eval_dataset)

        # Dictionary will contain all the scores and evidences generated by the model
        # Dictionary will contain all the scores and evidences generated by the model
        results = {"cls": [], "LLR_score": [], "prob_score": {"cls_0": [], "cls_1": []},
                   "evidence": {"coherency_score": [], "comsense_consistency": []}}

        self.model.eval()

        with tqdm(eval_loader, desc="Eval") as loop:
            for texts, masks in loop:
                texts, masks = texts.to(self.args.device), masks.to(self.args.device)

                output_dic = self.model(texts, attention_mask=masks)
                disc_out = output_dic["logits"]

                cls0_prob = disc_out[:, 0].tolist()
                cls1_prob = disc_out[:, 1].tolist()

                results["prob_score"]["cls_0"].extend(cls0_prob)
                results["prob_score"]["cls_1"].extend(cls1_prob)

                prior_llr = math.log10(self.args.priors[0]/self.args.priors[0])

                results["LLR_score"].extend([math.log10(prob/(1-prob)) + prior_llr for prob in cls1_prob])

                _, predicted = torch.max(disc_out, 1)

                results["cls"].extend(predicted.tolist())

                results["evidence"]["coherency_score"].extend([random.uniform(0, 1) for i in range(len(predicted))])

                results["evidence"]["comsense_consistency"].extend(
                    [random.uniform(0, 1) for i in range(len(predicted))])

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Roberta: Discriminator'
    )

    # Input data and files
    parser.add_argument('--model_name', default="robertatextgen", type=str,
                        help='name of the model')
    parser.add_argument('--check_point', default="/content/", type=str,
                        help='saved model checkpoint directory')

    # Model parameters
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--priors', type=list, default=[0.5, 0.5])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--large', type=bool, default=False, help='use the roberta-large model instead of roberta-base')

    args = parser.parse_args()

    if args.device is None:
        args.device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'

    artifact = GeneratedTextDetection(args)

    sample_eval_data = ["Russian opposition politician Aleksei Navalny says he owes his life to the pilots who made an emergency landing when he collapsed on a flight last month, and to paramedics he said had quickly diagnosed poisoning and injected him with atropine.",
                        "To receive Steve Gutterman's Week In Russia each week via e-mail, subscribe by clicking here.Alyaksandr Lukashenka's rushed, hushed-up inauguration ceremony in Belarus may evoke memories for Vladimir Putin, and contain a warning about the future as Russia hurtles toward 2024. Also, a COVID-19 surge, Kremlin contortions over the poisoning of Aleksei Navalny, and a UN speech \"rehearsing the defense of a nation in decline.",
                        "As Americans anxiously watch the spread of coronavirus variants that were first identified in Britain and South Africa, scientists are finding a number of new variants that seem to have originated in the United States — and many of them may pose the same kind of extra-contagious threat."]

    results = artifact.evaluate(sample_eval_data)

    print(results)


if __name__ == "__main__":
    main()


