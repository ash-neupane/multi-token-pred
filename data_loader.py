import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import List
import random

class ShortJokesDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, block_size: int, shuffle: bool = False) -> None:
        """
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.indices = list(range(0, len(self.dataset)))
    
    def __iter__(self) -> List[str]:
        """
        """
        if self.shuffle:
            random.shuffle(self.indices)
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx+self.batch_size]
            batch_jokes = torch.tensor(
                sum(
                    [self.dataset[i] for i in batch_indices], 
                    [self.dataset.bos_encoded] * (self.block_size)
                ) + 
                [self.dataset.bos_encoded,], 
                dtype=torch.int32
            )
            context = torch.empty((len(batch_jokes) - self.block_size, self.block_size), dtype=torch.int32) # N x block_size
            for i in range(0, len(batch_jokes)-self.block_size):
                context[i] = batch_jokes[i:i+self.block_size]                
            targets = batch_jokes[self.block_size:]
            yield context, targets

class ShortJokes(Dataset):
    def __init__(self, data_file="data/shortjokes.csv") -> None:
        self.data = [self.clean(j) for j in pd.read_csv(data_file)["Joke"]]
        self.bos_token = "<s>"
        self.bos_encoded = 0
        self._vocab = [self.bos_token,] + sorted(list(set("".join(self.data))))
        self._vocab_size = len(self._vocab)
        self._size = len(self.data)

        self.stoi = {ch: i for i, ch in enumerate(self._vocab)}
        self.itos = {i: ch for i, ch in enumerate(self._vocab)}

    def clean(self, joke: str) -> str:
        joke = joke.lower().strip()
        joke = joke.replace("\x08", "")
        joke = joke.replace("\x10", "")
        return joke
    
    def encode(self, joke: str) -> List[int]:
        joke = list(joke)
        return [self.stoi[ch] for ch in joke]

    def decode(self, encoded_joke: List[int]) -> str:        
        if isinstance(encoded_joke, int) or (isinstance(encoded_joke, torch.Tensor) and encoded_joke.ndim == 0):
            encoded_joke = [encoded_joke,]
        idx = [i.item() if isinstance(i, torch.Tensor) else i for i in encoded_joke]
        return "".join([self.itos[i] for i in idx])

    @property
    def vocab_size(self):
        return self._vocab_size 
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, i):
        return self.encode(self.data[i])

    def print_dataset_stats(self):
        print("\n-----Dataset Stats-------------")
        print("Number of jokes: ", len(self.data))
        print("Longest: ", max([len(j) for j in self.data]), "Shortest: ", min([len(j) for j in self.data]))
        chars = list("".join(self.data))
        char_counts = {}
        for ch in chars:
            char_counts[ch] = char_counts.get(ch, 0) + 1
        print("Counts of each char: ", sorted(char_counts.items(), key = lambda kv: -kv[1]))
        total_count = sum([char_counts[k] for k in char_counts.keys()])
        print("Probability of each char: ", list(zip(char_counts.keys(), [char_counts[k]/total_count for k in char_counts.keys()])))
        unique_chars = sorted(list(set(chars)))
        print("Unique: ", len(unique_chars), unique_chars)
        print("------------------------\n")
