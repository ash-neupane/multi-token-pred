import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import List
import random
from tqdm import tqdm
import tiktoken

class ShortJokesDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, block_size: int) -> None:
        """
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.data_len = len(self.dataset)
    
    def __iter__(self) -> List[str]:
        """
        """
        start_ix = torch.randint(self.data_len-self.block_size, (self.batch_size,))
        context = torch.stack(
            [self.dataset[i:i+self.block_size] for i in start_ix]
        )
        targets = torch.stack(
            [self.dataset[i+1:i+1+self.block_size] for i in start_ix]
        )

        yield context, targets

class ShortJokes(Dataset):
    def __init__(self, tokenization="bpe", max_tokens=6e6, data_file="data/shortjokes.csv") -> None:
        self.data = [self.clean(j) for j in pd.read_csv(data_file)["Joke"]]
        self.bos_token = "<s>"
        self.bos_encoded = 0
        
        if tokenization == "char":
            self._vocab = [self.bos_token,] + sorted(list(set("".join(self.data))))
            self.vocab_size = len(self._vocab)
            self.stoi = {ch: i for i, ch in enumerate(self._vocab)}
            self.itos = {i: ch for i, ch in enumerate(self._vocab)}
            
            encoded_data = torch.tensor([], dtype=torch.int32)
            for joke in tqdm(self.data):
                encoded_joke = torch.tensor([self.bos_encoded,] + self.encode(joke), dtype=torch.int32)
                encoded_data = torch.cat((encoded_data, encoded_joke))
                if encoded_data.size(0) > max_tokens:
                    break
            
        elif tokenization == "bpe":
            enc = tiktoken.get_encoding("gpt2")
            self.vocab_size = 50257 # TODO: get this automatically 

            encoded_data = torch.tensor([], dtype=torch.int32)
            for joke in tqdm(self.data):
                encoded_joke = torch.tensor([self.bos_encoded,] + enc.encode_ordinary(joke), dtype=torch.int32)
                encoded_data = torch.cat((encoded_data, encoded_joke))
                if encoded_data.size(0) > max_tokens:
                    break
        print("encoded data size", encoded_data.size())
        self.data = encoded_data

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
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

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

if __name__ == "__main__":
    ShortJokes(tokenization="bpe")