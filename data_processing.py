from torch.utils.data import Dataset
import torch
from collections import defaultdict, Counter
import pickle
from torchtext.vocab import Vocab
from tqdm import tqdm


def get_data(filename, samples=-1, truncation=-1):
    f = open(filename, mode='r', encoding='utf-8')
    sentences = []
    counter = 0
    for x, row in enumerate(tqdm(f)):
        sentence = row.split()
        end_idx = min(truncation, len(sentence)-1)
        sentences.append(sentence[:end_idx])
        counter += 1
        if counter == samples:
            break
    f.close()
    return sentences


# expects list of lists of words to generate a dictionary word2idx, and array id2word
def generate_vocab(sentences, unk_thresh=1):
    d = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            d[word] += 1

    word2idx = {'<UNK>': 0,
                '<SOS>': 1,
                '<EOS>': 2,
                '<PAD>': 3}
    id2word = ['<UNK>', '<SOS>', '<EOS>', '<PAD>']

    for key, value in d.items():
        if value >= unk_thresh:
            idx = len(word2idx)
            word2idx[key] = idx
            id2word.append(key)

    return word2idx, id2word


def build_vocabulary(sentences, vocab_size: int, vocab_file: str = None, load: bool = False):
    if load and vocab_file:
        with open(vocab_file, 'rb') as file:
            vocab = pickle.load(file)
            return vocab

    counts = Counter()

    for x, sentence in enumerate(tqdm(sentences)):
        counts.update(sentence)

    vocab = Vocab(
        counter=counts,
        max_size=vocab_size,
        specials=('<UNK>', '<SOS>', '<EOS>', '<PAD>'),
    )

    if vocab_file:
        with open(vocab_file, "wb") as v:
            pickle.dump(vocab, v)
    return vocab


def prepare_batch(batch, pad_idx):
    # batch.size = [batch len, (sources, targets)]
    sources, targets = zip(*batch)
    max_src_len = max(len(t) for t in sources)
    max_tgt_len = max(len(t) for t in targets)
    batch_size = len(sources)

    sources_tensor = torch.full((max_src_len, batch_size), pad_idx, dtype=torch.long)
    targets_tensor = torch.full((max_tgt_len, batch_size), pad_idx, dtype=torch.long)

    for x, sentence in enumerate(sources):
        for y, word in enumerate(sentence):
            sources_tensor[y][x] = word
    for x, sentence in enumerate(targets):
        for y, word in enumerate(sentence):
            targets_tensor[y][x] = word

    return sources_tensor, targets_tensor


class TrainDataset(Dataset):
    def __init__(self, sources, targets, vocab, unk_idx, sos_idx, eos_idx):
        self.sources = sources
        self.targets = targets
        self.vocab = vocab
        self.unk_idk = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __getitem__(self, item):
        source = self.sources[item]
        target = self.targets[item]
        idx_source = [self.sos_idx]
        idx_target = [self.sos_idx]
        for word in source:
            idx_source.append(self.vocab.get(word, self.unk_idk))
        for word in target:
            idx_target.append(self.vocab.get(word, self.unk_idk))
        idx_source.append(self.eos_idx)
        idx_target.append(self.eos_idx)
        return idx_source, idx_target

    def __len__(self):
        return len(self.sources)


class TestDataset(Dataset):
    def __init__(self, sources, vocab, unk_idx, sos_idx, eos_idx):
        self.sources = sources
        self.vocab = vocab
        self.unk_idk = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __getitem__(self, item):
        source = self.sources[item]
        idx_source = [self.sos_idx]
        for word in source:
            idx_source.append(self.vocab.get(word, self.unk_idk))
        idx_source.append(self.eos_idx)
        return idx_source

    def __len__(self):
        return len(self.sources)

# remaining work:
# design data loader
# design collate function that properly pads the sequences to their respective max length
# Q: do these need to be the same length for the encoder/decoder? can they be different for each

# need to build a vocab generate, maybe use Nilay's approach instead of sklearn
# shared vocab between source and target


