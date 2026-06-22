import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

#1 데이터 파이프라인
DATA_FILE = "wealth.txt"
if not Path(DATA_FILE).exists():
    raise FileNotFoundError(f"'{DATA_FILE}' 파일을 찾을 수 없습니다.")

text = open(DATA_FILE, "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long).to(device)

block_size = 64
sequences = torch.stack([data[i:i+block_size+1] for i in range(len(data)-block_size)])

#2 GPT 블록
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ self.value(x)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = Head(n_embd, n_embd, block_size) 
        self.ffn = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd))
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, x):
        B, T = x.shape
        x = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        x = self.blocks(x)
        return self.head(self.ln_f(x))

model = TinyGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#3 학습 루틴
if __name__ == '__main__':
    TOTAL_EPOCHS = 2000
    BATCH_SIZE = 2048
    
    print(f"학습 시작 ({device}) - 국부론 학습 중")
    model.train()
    
    pbar = tqdm(range(TOTAL_EPOCHS), desc="학습 진행도")
    for epoch in pbar:
        ix = torch.randint(0, len(sequences), (BATCH_SIZE,), device=device)
        batch = sequences[ix]
        xb, yb = batch[:, :-1], batch[:, 1:]
        
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 화면에 loss 값을 띄워주는 역할
        pbar.set_postfix_str(f"loss={loss.item():.4f}")

        # 500번마다 터미널에 Loss 기록 남김
        if epoch % 500 == 0:
            print(f"\n👉 [에포크 {epoch}/2000] 현재 Loss: {loss.item():.4f}")

    #4 생성
    model.eval()
    start_str = "market "
    idx = torch.tensor([stoi[s] for s in start_str], dtype=torch.long, device=device).view(1, -1)
    out = list(start_str)
    for _ in range(1500):
        logits = model(idx[:, -block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_idx.view(1, 1)], dim=1)
        out.append(itos[next_idx.item()])
        
    with open("result_for_professor.txt", "w", encoding="utf-8") as f:
        f.write("".join(out))
    print("\n완료! 결과물을 확인하세요.")