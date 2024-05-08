import random
import torch
from torch.nn import functional as F
from data_loader import ShortJokesDataLoader, ShortJokes
import time

batch_size = 10
n_steps = 0
max_iter = 5e5
block_size = 24
n_embed = 16
hidden_dim = 100
PRINT_FREQUENCY = 5e3

jokes_dataset = ShortJokes()
jokes_dataset.print_dataset_stats()

jokes_loader = ShortJokesDataLoader(jokes_dataset, batch_size, block_size, shuffle=True)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)

# print("BEGIN TRAINING")
# epochs = 1
# avg_time_per_iter = 0
# global_start_time = time.perf_counter()
# for _ in range(epochs):
#     for context, target in jokes_loader:
#         start_time = time.perf_counter()
#         n_steps += 1
#         model.zero_grad()
#         logits, loss = model(context, targets = target)
#         loss.backward()
#         optimizer.step()
#         end_time = time.perf_counter()
#         avg_time_per_iter = 0.99 * avg_time_per_iter + 0.01 * (end_time - start_time) 
#         if n_steps >= max_iter:
#             break
#         elif n_steps % PRINT_FREQUENCY == 1:
#             print(f"[{n_steps}] batches. Loss: {loss}. Avg time per iteration: {avg_time_per_iter:.4f} sec")
# total_elapsed_time = time.perf_counter() - global_start_time
# print(f"Completed training after [{n_steps}] batches. Loss: {loss}. Avg time per iteration: {avg_time_per_iter:.4f} sec")
# print(f"Total training time: {(total_elapsed_time)/(60.0)} mins")
# torch.save(model.state_dict(), f"model/{type(model).__name__.lower()}.pth")

# print("-------------------------------------------\nUNIFORM INITIALIZATION\n")
# print("uniform probability: ", 1/69)
# print("expected loss: ", -torch.tensor(1/69.0).log())
# print("initialized logits in", logits.min().item(), "to", logits.max().item())

