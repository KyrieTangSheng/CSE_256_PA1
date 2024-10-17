from collections import Counter, defaultdict
import re

class BPE_Tokenizer:
    def __init__(self, dataset, vocab_size_limit):
        self.dataset = dataset
        self.vocab_size_limit = vocab_size_limit
        self.vocab = self.get_vocab()
        self.merges = {}
        self.train()
        self.subword_to_index, self.index_to_subword = self.build_index_vocab()

    def get_vocab(self):
        self.vocab = defaultdict(int)
        for sentence in self.dataset:
            for word in sentence.words:
                self.vocab[' '.join(list(word)) + ' </w>'] += 1
        return self.vocab
    
    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in self.vocab:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = self.vocab[word]
        self.vocab = v_out

    def train(self):
        while len(self.vocab) < self.vocab_size_limit:
            pairs = self.get_stats()
            best = max(pairs, key=pairs.get)
            self.merge_vocab(best)
            self.merges[best] = ''.join(best)
    
    def get_vocab_dict(self):
        return self.vocab
    
    def build_index_vocab(self):
        # Assign an index to each unique subword in the vocabulary
        subword_to_index = {"<unk>": 0}  # Reserve index 0 for <unk>
        for idx, subword in enumerate(self.vocab.keys(), start=1):
            subword_to_index[subword.replace(" </w>", "")] = idx  # Remove end-of-word token for consistency
        index_to_subword = {}
        for k,v in subword_to_index.items():
            index_to_subword[v] = k
        return subword_to_index, index_to_subword
    
    def tokenize(self, list_of_words):
        tokenized_indices = []
        for word in list_of_words:
            word = ' '.join(list(word)) + ' </w>'
            tokenized_word = []
            while word:
                match_found = False
                for token in sorted(self.vocab.keys(), key=len, reverse=True):
                    if word.startswith(token):
                        tokenized_word.append(token)
                        word = word[len(token):].lstrip()
                        match_found = True
                        break
                if not match_found:
                    # Handle the case where no token matches
                    tokenized_word.append(word.split()[0])
                    word = ' '.join(word.split()[1:])
            tokenized_indices.extend([self.subword_to_index.get(token.replace(" </w>", ""), 0) for token in tokenized_word])
        return tokenized_indices

    
def test_BPE_Tokenizer():
    from sentiment_data import read_sentiment_examples
    # test the tokenizer
    dataset = read_sentiment_examples("data/train.txt")
    tokenizer = BPE_Tokenizer(dataset=dataset, vocab_size_limit=10000)
    tokenized_index = tokenizer.tokenize(dataset[0].words)
    print(tokenized_index)


import torch.nn
from torch import nn
from torch.utils.data import Dataset
from sentiment_data import read_word_embeddings, read_sentiment_examples
class SentimentDataset_BPE_WordEmbedding(Dataset):
    def __init__(self, examples_file, BPE_tokenizer):
        #get examples and labels
        self.examples = read_sentiment_examples(examples_file)
        self.tokenizer = BPE_tokenizer
        # Calculate max_length based on tokenized sequences
        self.word_indices = []
        max_length = 0
        count = 0
        for ex in self.examples:
            tokenized_indices = self.tokenizer.tokenize(ex.words)
            self.word_indices.append(tokenized_indices)
            max_length = max(max_length, len(tokenized_indices))
            count += 1
            if count % 1000 == 0:
                print(f"processed {count} examples...")
        
        # Pad sequences to max_length
        self.word_indices = [torch.tensor(indices + [0] * (max_length - len(indices))) for indices in self.word_indices]
        self.labels = torch.tensor([ex.label for ex in self.examples])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.word_indices[index], self.labels[index]
    

class DAN2Model_BPE(nn.Module):
    def __init__(self, embedding_dim=50, hidden_size=100, vocab_size=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(self.embedding_dim, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.int()
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.softmax(x)
        return x