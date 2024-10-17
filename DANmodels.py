import torch.nn
from torch import nn
from torch.utils.data import Dataset
from sentiment_data import read_word_embeddings, read_sentiment_examples

class SentimentDatasetWordEmbedding(Dataset):
    def __init__(self, examples_file, embeddings_file):
        #get examples and labels
        self.examples = read_sentiment_examples(examples_file)
        word_embeddings = read_word_embeddings(embeddings_file)
        max_length = max(len(ex.words) for ex in self.examples)
        self.word_indices = []
        for ex in self.examples:
            self.word_indices.append(torch.tensor([word_embeddings.word_indexer.index_of(word) 
                                                    if word_embeddings.word_indexer.index_of(word) != -1 else 1
                                                    for word in ex.words] +
                                                  [word_embeddings.word_indexer.index_of('PAD')] * (max_length - len(ex.words)) ))
        self.labels = torch.tensor([ex.label for ex in self.examples])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.word_indices[index], self.labels[index]

class DAN2Model_GloVe(nn.Module):
    def __init__(self, embedding_dim, hidden_size=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        embeddings_file = f'data/glove.6B.{embedding_dim}d-relativized.txt'
        self.embedding_class = read_word_embeddings(embeddings_file)
        self.embedding_layer = self.embedding_class.get_initialized_embedding_layer(frozen=False)
        self.output_size = 2
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(embedding_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.int()
        embeddings = self.embedding_layer(x)
        # Compute the average embedding
        hidden_states = torch.mean(embeddings, dim=1)
        hidden_states = self.layer1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer2(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.softmax(hidden_states)
    
    
class DAN3Model_GloVe(nn.Module):
    def __init__(self, embedding_dim, hidden_size=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        embeddings_file = f'data/glove.6B.{embedding_dim}d-relativized.txt'
        self.embedding_class = read_word_embeddings(embeddings_file)
        self.embedding_layer = self.embedding_class.get_initialized_embedding_layer(frozen=False)
        self.output_size = 2
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(embedding_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.int()
        embeddings = self.embedding_layer(x)
        # Compute the average embedding
        hidden_states = torch.mean(embeddings, dim=1)
        hidden_states = self.layer1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer2(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer3(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.softmax(hidden_states)
    

class DAN2Model_Random(nn.Module):
    def __init__(self, embedding_dim, hidden_size=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_size = 2
        embeddings_file = f'data/glove.6B.{embedding_dim}d-relativized.txt'
        self.embedding_class = read_word_embeddings(embeddings_file)
        self.vocab_size = self.embedding_class.vectors.shape[0]
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.layer1 = nn.Linear(self.embedding_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = x.int()
        embeddings = self.embedding_layer(x)
        hidden_states = torch.mean(embeddings, dim=1)
        hidden_states = self.layer1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer2(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.softmax(hidden_states)

class DAN3Model_Random(nn.Module):
    def __init__(self, embedding_dim, hidden_size=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_size = 2
        embeddings_file = f'data/glove.6B.{embedding_dim}d-relativized.txt'
        self.embedding_class = read_word_embeddings(embeddings_file)
        self.vocab_size = self.embedding_class.vectors.shape[0]
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)   
        self.layer1 = nn.Linear(self.embedding_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.ReLU()

    def forward(self, x):   
        x = x.int()
        embeddings = self.embedding_layer(x)
        hidden_states = torch.mean(embeddings, dim=1)
        hidden_states = self.layer1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer2(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer3(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.softmax(hidden_states)

