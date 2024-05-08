import random
import torch
from torch.nn import functional as F
from data_loader import ShortJokesDataLoader, ShortJokes
from model import ModelConfig, VanillaTransformer
import time

device = "cpu"
batch_size = 1
max_iter = 1e4
PRINT_FREQUENCY = 5e2

jokes_dataset = ShortJokes()
#jokes_dataset.print_dataset_stats()

block_size = 32
jokes_loader = ShortJokesDataLoader(jokes_dataset, batch_size, block_size, shuffle=True)

model = VanillaTransformer(
    ModelConfig(
        vocab_size = jokes_dataset.vocab_size,  
        block_size = block_size,
        n_layer = 4,
        n_heads = 4,
        d_model = 32 * 4
    )
)
model.to(device)
print("Num parameters: ", model.get_num_params())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.99), eps=1e-8)

print("BEGIN TRAINING")
n_steps = 0
epochs = 1
avg_time_per_iter = 0
global_start_time = time.perf_counter()
for _ in range(epochs):
    for context, target in jokes_loader:
        context.to(device)
        target.to(device)
        # print("ctx: ", context.size(), context.dtype, context.device)
        # print("target: ", target.size(), target.dtype,target.device)
        start_time = time.perf_counter()
        n_steps += 1
        model.zero_grad()
        logits, loss = model(context, targets = target)
        loss.backward()
        optimizer.step()
        end_time = time.perf_counter()
        avg_time_per_iter = 0.99 * avg_time_per_iter + 0.01 * (end_time - start_time) 
        if n_steps >= max_iter:
            break
        elif n_steps % PRINT_FREQUENCY == 1:
            print(f"[{n_steps}] batches. Loss: {loss}. Avg time per iteration: {avg_time_per_iter:.4f} sec")
total_elapsed_time = time.perf_counter() - global_start_time
print(f"Completed training after [{n_steps}] batches. Loss: {loss}. Avg time per iteration: {avg_time_per_iter:.4f} sec")
print(f"Total training time: {(total_elapsed_time)/(60.0)} mins")
torch.save(model.state_dict(), f"model/{type(model).__name__.lower()}.pth")

# print("-------------------------------------------\nUNIFORM INITIALIZATION\n")
# print("uniform probability: ", 1/70)
# print("expected loss: ", -torch.tensor(1/70.0).log())
# print("initialized logits in", logits.min().item(), "to", logits.max().item())

