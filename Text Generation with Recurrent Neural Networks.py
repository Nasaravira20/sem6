import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


# for data download
# wget -O shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt

# Hyperparameters
seq_length = 100
hidden_size = 256
lr = 0.002
epochs = 1
batch_size = 512
temperature = 0.8

# Load and preprocess text
with open('data/shakespeare.txt', 'r') as f:
    text = f.read().lower()

# Read only 5% of the text length
text = text[:int(len(text) * 0.05)]

chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
vocab_size = len(chars)

# Custom Dataset
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.text) - self.seq_length - 1

    def __getitem__(self, idx):
        seq_in = self.text[idx:idx + self.seq_length]
        seq_out = self.text[idx + 1:idx + self.seq_length + 1]
        input_seq = torch.zeros(self.seq_length, vocab_size)
        target_seq = torch.tensor([char_to_idx[char] for char in seq_out])

        for i, char in enumerate(seq_in):
            input_seq[i][char_to_idx[char]] = 1.0
        return input_seq, target_seq

dataset = ShakespeareDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=next(self.parameters()).device)

model = CharRNN(vocab_size, hidden_size, vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_inputs, batch_targets in dataloader:
        batch_size = batch_inputs.size(0)
        hidden = model.init_hidden(batch_size)

        optimizer.zero_grad()
        output, hidden = model(batch_inputs, hidden)
        output = output.view(-1, vocab_size)
        batch_targets = batch_targets.view(-1)

        loss = loss_fn(output, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Text generation
def generate_text(model, seed, num_chars, temperature=1.0):
    model.eval()
    input_seq = torch.zeros(1, len(seed), vocab_size, device=next(model.parameters()).device)
    for i, char in enumerate(seed):
        input_seq[0, i, char_to_idx.get(char, char_to_idx[' '])] = 1.0

    hidden = model.init_hidden(1)  # Batch size is 1 for text generation
    generated = seed

    for _ in range(num_chars):
        output, hidden = model(input_seq, hidden)
        output = output[0, -1] / temperature
        probs = torch.softmax(output, dim=0).detach().cpu().numpy()
        char_idx = np.random.choice(len(chars), p=probs)
        generated += idx_to_char[char_idx]

        # Prepare the next input
        next_input = torch.zeros(1, 1, vocab_size, device=next(model.parameters()).device)
        next_input[0, 0, char_idx] = 1.0
        input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)

    return generated


# Example generation
seed_text = "shall i compare thee to a summer's day"
print("\nGenerated Text:\n")
print(generate_text(model, seed_text, 500, temperature))
