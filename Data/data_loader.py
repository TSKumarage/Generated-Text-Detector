from torchtext import data, datasets
from torchtext.vocab import GloVe
import re
import spacy
import en_core_web_sm
import pickle


class TrainDataset:
    """
       Dataset class
       - Create an iterator objects for training and eval data
       - Populate vocab
    """

    def __init__(self, emb_dim=50, mbsize=32, custom_data=False, eval=False, train_data_path="", eval_data_file="",
                 checkpoint_path=""):

        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize=self._tokenizer,
                               fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        self.MAX_CHARS = 20000
        self.NLP = en_core_web_sm.load()

        if not eval:

            # Only take sentences with length <= 15
            f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'

            if custom_data:

                # create tuples representing the columns
                fields = [
                    (None, None),
                    ('text', self.TEXT),
                    (None, None),
                    (None, None),
                    ('label', self.LABEL)
                ]

                # load the dataset in json format
                train_data, validation_data, test_data = data.TabularDataset.splits(
                    path=train_data_path,
                    train='train_data.csv',
                    validation='validation_data.csv',
                    test='test_data.csv',
                    format='csv',
                    fields=fields,
                    skip_header=True
                )

            else:
                train_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)

                train_data, validation_data = train_data.split()

            self.TEXT.build_vocab(train_data, vectors=GloVe('6B', dim=emb_dim))
            self.LABEL.build_vocab(train_data)

            self.n_vocab = len(self.TEXT.vocab.itos)
            self.emb_dim = emb_dim

            self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
                (train_data, validation_data, test_data), batch_size=mbsize, device=-1, sort_key=lambda x: len(x.text),
                shuffle=True, repeat=True
            )

            self.train_loader = self.train_iter
            self.test_loader = self.test_iter
            self.validation_loader = self.val_iter

            self.train_iter = iter(self.train_iter)
            self.val_iter = iter(self.val_iter)
            self.test_iter = iter(self.test_iter)

        else:

            self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize=self._tokenizer,
                                   fix_length=16)
            self.TEXT.vocab = self._get_from_checkpoint(checkpoint_path)

            self.n_vocab = len(self.TEXT.vocab.itos)

            fields = [
                ('text', self.TEXT)
            ]

            # load the dataset in json format
            test_data = data.TabularDataset(
                path=eval_data_file,
                format='csv',
                fields=fields,
                skip_header=True
            )

            self.test_iter = data.BucketIterator(
                test_data,
                batch_size=mbsize,
                device=-1,
                sort_key=lambda x: len(x.text),
                shuffle=False,
                repeat=False
            )

            self.test_loader = self.test_iter
            self.test_iter = iter(self.test_iter)

    def _get_from_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, "rb")as f:
            vocab = pickle.load(f)

            return vocab

    def _tokenizer(self, review):

        line = str(review).replace("\n", "")
        line = line.replace("\r", "")
        line = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', line,
                      flags=re.MULTILINE)

        review = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ",
            line)
        review = re.sub(r"[ ]+", " ", review)
        review = re.sub(r"\!+", "!", review)
        review = re.sub(r"\,+", ",", review)
        review = re.sub(r"\?+", "?", review)
        if (len(review) > self.MAX_CHARS):
            review = review[:self.MAX_CHARS]
        return [x.text for x in self.NLP.tokenizer(review) if x.text != " "]

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.to(device="cuda"), batch.label.to(device="cuda")

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.to(device="cuda"), batch.label.to(device="cuda")

        return batch.text, batch.label

    def next_test_batch(self, gpu=False):
        batch = next(self.test_iter)

        if gpu:
            return batch.text.to(device="cuda"), batch.label.to(device="cuda")

        return batch.text, batch.label

    def next_eval_batch(self, gpu=False):
        batch = next(self.test_iter)

        if gpu:
            return batch.text.to(device="cuda")

        return batch.text

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

    def save_vocab(self, path):
        with open(path + "/vocab.pkl", "wb")as f:
            pickle.dump(self.TEXT.vocab, f)

