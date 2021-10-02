import math
import torch
import argparse
from tqdm import tqdm

import random
import time

from torch.utils.data import DataLoader
from common.dataset import EncodeEvalData

from transformers import *
from common.detector import RobertaForTextGenClassification
from common.attributor import RobertaForGeneratorAttribution


class GeneratedTextDetection:
    """
    Artifact class
    """

    def __init__(self, args):
        torch.manual_seed(int(time.time()))

        self.args = args

        # Load the model from checkpoints
        self.init_dict = self._init_detector()

    def _init_detector(self):

        init_dict = {"kn_model": None, "kn_tokenizer": None,
                    "unk_model": None, "unk_tokenizer": None,
                   "attr_model": None, "attr_tokenizer": None, }

        if self.args.init_method == "individual":
            model_name = 'roberta-large' if self.args.kn_large else 'roberta-base'
            tokenization_utils.logger.setLevel('ERROR')
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForTextGenClassification.from_pretrained(model_name).to(self.args.device)

            # Load the model from checkpoints
            if self.args.device == "cpu":
                model.load_state_dict(torch.load((self.args.check_point + '{}.pt').format(self.args.known_model_name),
                                                 map_location='cpu')['model_state_dict'])
            else:
                model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.known_model_name))['model_state_dict'])

            init_dict["kn_model"] = model
            init_dict["kn_tokenizer"] = tokenizer

        elif self.args.init_method == "highest_confidence":

            kn_model_name = 'roberta-large' if self.args.kn_large else 'roberta-base'
            tokenization_utils.logger.setLevel('ERROR')
            kn_tokenizer = RobertaTokenizer.from_pretrained(kn_model_name)
            kn_model = RobertaForTextGenClassification.from_pretrained(kn_model_name).to(self.args.device)

            # Load the model from checkpoints
            if self.args.device == "cpu":
                kn_model.load_state_dict(torch.load((self.args.check_point + '{}.pt').format(self.args.known_model_name),
                                                 map_location='cpu')['model_state_dict'])
            else:
                kn_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.known_model_name))[
                        'model_state_dict'])

            init_dict["kn_model"] = kn_model
            init_dict["kn_tokenizer"] = kn_tokenizer

            unk_model_name = 'roberta-large' if self.args.unk_large else 'roberta-base'
            tokenization_utils.logger.setLevel('ERROR')
            unk_tokenizer = RobertaTokenizer.from_pretrained(unk_model_name)
            unk_model = RobertaForTextGenClassification.from_pretrained(unk_model_name).to(self.args.device)

            # Load the model from checkpoints
            if self.args.device == "cpu":
                unk_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.unknown_model_name),
                               map_location='cpu')['model_state_dict'])
            else:
                unk_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.unknown_model_name))[
                        'model_state_dict'])

            init_dict["unk_model"] = unk_model
            init_dict["unk_tokenizer"] = unk_tokenizer

        elif self.args.init_method == "attribution_switch":

            kn_model_name = 'roberta-large' if self.args.kn_large else 'roberta-base'
            tokenization_utils.logger.setLevel('ERROR')
            kn_tokenizer = RobertaTokenizer.from_pretrained(kn_model_name)
            kn_model = RobertaForTextGenClassification.from_pretrained(kn_model_name).to(self.args.device)

            # Load the model from checkpoints
            if self.args.device == "cpu":
                kn_model.load_state_dict(torch.load((self.args.check_point + '{}.pt').format(self.args.known_model_name),
                                                 map_location='cpu')['model_state_dict'])
            else:
                kn_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.known_model_name))[
                        'model_state_dict'])

            init_dict["kn_model"] = kn_model
            init_dict["kn_tokenizer"] = kn_tokenizer

            unk_model_name = 'roberta-large' if self.args.unk_large else 'roberta-base'
            tokenization_utils.logger.setLevel('ERROR')
            unk_tokenizer = RobertaTokenizer.from_pretrained(unk_model_name)
            unk_model = RobertaForTextGenClassification.from_pretrained(unk_model_name).to(self.args.device)

            # Load the model from checkpoints
            if self.args.device == "cpu":
                unk_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.unknown_model_name),
                               map_location='cpu')['model_state_dict'])
            else:
                unk_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.unknown_model_name))[
                        'model_state_dict'])

            init_dict["unk_model"] = unk_model
            init_dict["unk_tokenizer"] = unk_tokenizer

            attr_model_name = 'roberta-large' if self.args.attr_large else 'roberta-base'
            tokenization_utils.logger.setLevel('ERROR')
            attr_tokenizer = RobertaTokenizer.from_pretrained(attr_model_name)
            attr_model = RobertaForGeneratorAttribution.from_pretrained(attr_model_name).to(self.args.device)

            # Load the model from checkpoints
            if self.args.device == "cpu":
                attr_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.attribution_model_name),
                               map_location='cpu')['model_state_dict'])
            else:
                attr_model.load_state_dict(
                    torch.load((self.args.check_point + '{}.pt').format(self.args.attribution_model_name))[
                        'model_state_dict'])

            init_dict["attr_model"] = attr_model
            init_dict["attr_tokenizer"] = attr_tokenizer

        return init_dict

    def evaluate(self, input_text):
        """
           Method that runs the evaluation and generate scores and evidence
        """

        # Encapsulate the inputs
        eval_dataset = EncodeEvalData(input_text, self.init_dict["kn_tokenizer"], self.args.max_sequence_length)
        eval_loader = DataLoader(eval_dataset)

        # Dictionary will contain all the scores and evidences generated by the model
        results = {"cls": [], "LLR_score": [], "prob_score": {"cls_0": [], "cls_1": []}}

        # Set eval mode
        if self.args.init_method == "individual":
            self.init_dict["kn_model"].eval()

        elif self.args.init_method == "highest_confidence":
            self.init_dict["kn_model"].eval()
            self.init_dict["unk_model"].eval()

        elif self.args.init_method == "attribution_switch":
            self.init_dict["kn_model"].eval()
            self.init_dict["unk_model"].eval()
            self.init_dict["attr_model"].eval()

        with torch.no_grad():
            with tqdm(eval_loader, desc="Eval") as loop:
                for texts, masks in loop:
                    texts, masks = texts.to(self.args.device), masks.to(self.args.device)

                    if self.args.init_method == "individual":
                        # Individual model take care all the probes
                        output_dic = self.init_dict["kn_model"](texts, attention_mask=masks)
                        disc_out = output_dic["logits"]

                        cls0_prob = disc_out[:, 0].tolist()
                        cls1_prob = disc_out[:, 1].tolist()

                        results["prob_score"]["cls_0"].extend(cls0_prob)
                        results["prob_score"]["cls_1"].extend(cls1_prob)

                        prior_llr = math.log10(self.args.kn_priors[0]/self.args.kn_priors[0])

                        results["LLR_score"].extend([math.log10(prob/(1-prob)) + prior_llr for prob in cls1_prob])

                        _, predicted = torch.max(disc_out, 1)

                        results["cls"].extend(predicted.tolist())

                    elif self.args.init_method == "highest_confidence":
                        # Models trained on known and unknown generators: Model with highest confidence would take the probe

                        # Prediction from known generator trained detector
                        output_dic = self.init_dict["kn_model"](texts, attention_mask=masks)
                        kn_disc_out = output_dic["logits"]

                        output_dic = self.init_dict["unk_model"](texts, attention_mask=masks)
                        unk_disc_out = output_dic["logits"]

                        kn_cls1_prob = kn_disc_out[:, 1].tolist()
                        unk_cls1_prob = unk_disc_out[:, 1].tolist()

                        kn_prior_llr = math.log10(self.args.kn_priors[0] / self.args.kn_priors[0])
                        unk_prior_llr = math.log10(self.args.unk_priors[0] / self.args.unk_priors[0])

                        kn_llr = [math.log10(prob / (1 - prob)) + kn_prior_llr for prob in kn_cls1_prob]
                        unk_llr = [math.log10(prob / (1 - prob)) + unk_prior_llr for prob in unk_cls1_prob]

                        if list(map(abs, kn_llr)) > list(map(abs, unk_llr)):
                            results["prob_score"]["cls_0"].extend(kn_disc_out[:, 0].tolist())
                            results["prob_score"]["cls_1"].extend(kn_cls1_prob)

                            results["LLR_score"].extend(kn_llr)

                            _, predicted = torch.max(kn_disc_out, 1)

                            results["cls"].extend(predicted.tolist())

                        else:

                            results["prob_score"]["cls_0"].extend(unk_disc_out[:, 0].tolist())
                            results["prob_score"]["cls_1"].extend(unk_cls1_prob)

                            results["LLR_score"].extend(unk_llr)

                            _, predicted = torch.max(unk_disc_out, 1)

                            results["cls"].extend(predicted.tolist())

                    elif self.args.init_method == "attribution_switch":

                        # Generator attribution model would predict the given probes source -
                        # Then the corresponding detector will be used

                        output_dic = self.init_dict["attr_model"](texts, attention_mask=masks)
                        attr_disc_out = output_dic["logits"]

                        _, predicted_gen = torch.max(attr_disc_out, 1)

                        if predicted_gen[0] == 1:
                            output_dic = self.init_dict["kn_model"](texts, attention_mask=masks)
                            disc_out = output_dic["logits"]

                            cls0_prob = disc_out[:, 0].tolist()
                            cls1_prob = disc_out[:, 1].tolist()

                            results["prob_score"]["cls_0"].extend(cls0_prob)
                            results["prob_score"]["cls_1"].extend(cls1_prob)

                            prior_llr = math.log10(self.args.kn_priors[0] / self.args.kn_priors[0])

                            results["LLR_score"].extend([math.log10(prob / (1 - prob)) + prior_llr for prob in cls1_prob])

                            _, predicted = torch.max(disc_out, 1)

                            results["cls"].extend(predicted.tolist())

                        else:

                            output_dic = self.init_dict["unk_model"](texts, attention_mask=masks)
                            disc_out = output_dic["logits"]

                            cls0_prob = disc_out[:, 0].tolist()
                            cls1_prob = disc_out[:, 1].tolist()

                            results["prob_score"]["cls_0"].extend(cls0_prob)
                            results["prob_score"]["cls_1"].extend(cls1_prob)

                            prior_llr = math.log10(self.args.unk_priors[0] / self.args.unk_priors[0])

                            results["LLR_score"].extend([math.log10(prob / (1 - prob)) + prior_llr for prob in cls1_prob])

                            _, predicted = torch.max(disc_out, 1)

                            results["cls"].extend(predicted.tolist())

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generated Text: Discriminator'
    )

    # Input data and files
    parser.add_argument('--known_model_name', default="robertatextgen", type=str,
                        help='name of the known generator detector model')

    parser.add_argument('--unknown_model_name', default="unkrobertatextgen", type=str,
                        help='name of the unknown generator detector model')

    parser.add_argument('--attribution_model_name', default="robertagenattr", type=str,
                        help='name of the generator attribution model')

    parser.add_argument('--init_method', default="individual", type=str,
                        help='name of the generator attribution model')

    parser.add_argument('--check_point', default="/content/", type=str,
                        help='saved model checkpoint directory')

    # Model parameters
    parser.add_argument('--device', type=str, default=None)

    parser.add_argument('--kn_priors', type=list, default=[0.5, 0.5])
    parser.add_argument('--unk_priors', type=list, default=[0.5, 0.5])

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--kn_large', type=bool, default=False)
    parser.add_argument('--unk_large', type=bool, default=False)
    parser.add_argument('--attr_large', type=bool, default=False)

    args = parser.parse_args()

    if args.device is None:
        args.device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'

    artifact = GeneratedTextDetection(args)

    sample_eval_data = ["Russian opposition politician Aleksei Navalny says he owes his life to the pilots who made an emergency landing when he collapsed on a flight last month, and to paramedics he said had quickly diagnosed poisoning and injected him with atropine."]

    results = artifact.evaluate(sample_eval_data)

    print(results)


if __name__ == "__main__":
    main()


