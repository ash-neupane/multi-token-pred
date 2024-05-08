import torch
from torch.nn import functional as F
from model import ModelConfig, VanillaTransformer
from data_loader import ShortJokes

jokes_dataset = ShortJokes()
vocab_size = jokes_dataset.vocab_size
block_size = 32
d_model = 32 * 4
def sample_from_model(model, temp=0.2, max_len = 300):
    model.eval()
    context = torch.tensor([jokes_dataset.bos_encoded]*block_size, dtype=torch.long).view(1, block_size)
    sampled_ix = model.generate(context, max_len, temp=temp, top_k=None)[0, block_size:]
    sampled_str = jokes_dataset.decode(sampled_ix).split("<s>")[0]
    model.train()
    return sampled_str

model = VanillaTransformer(
    ModelConfig(
        vocab_size = jokes_dataset.vocab_size,  
        block_size = block_size,
        n_layer = 4,
        n_heads = 4,
        d_model = 32 * 4
    )
)
state_dict = torch.load("model/vanillatransformer.pth")
model.load_state_dict(state_dict)

for i in range(10):
    sampled = sample_from_model(model)
    print(f"{i+1}. {sampled}")
