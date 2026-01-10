import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
#define of Bigram

    
#hyperparameters
batch_size = 32
block_size = 4
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#======

torch.manual_seed(1337)

#prepare dataset
data_file_path = os.path.join(os.path.dirname(__file__),"input.txt")
if not os.path.exists(data_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'    
    with open (data_file_path,"w",encoding="utf-8") as file:
        file.write(requests.get(data_url).text())

with open(data_file_path,'r',encoding='utf-8') as file:
    data = file.read()

words = sorted(list(set(data)))
char_size = len(words)
itoc = {i:ch for i,ch in enumerate(words)}
ctoi = {ch:i for i,ch in enumerate(words)}
encode  = lambda s: [ctoi[ch] for ch in s]
decode = lambda l:"".join([itoc[i] for i in l])
data = torch.tensor(encode(data),dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer("tril",torch.trill(torch.ones(block_size,block_size)))
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei =q @ k.transpose(-2,-1)*head_size**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei = F.softmax(wei,dim=-1)
        out = wei@v
        return out
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(char_size,n_embd) 
        self.position_embedding_table =nn.Embedding(block_size,n_embd) 
        self.im_head= nn.Linear(n_embd,char_size)
        self.sa_head = Head(n_embd)
        
    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arrange(T,device=device))
        x= tok_emb+pos_emb
        x = self.sa_head(x)
        logits = self.im_head(tok_emb)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss            
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next= torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx



def get_batch(split):
    data =  train_data if split =="train" else test_data
    index= torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in index])
    y = torch.stack([data[i+1:i+block_size+1] for i in index])
    return x.to(device),y.to(device)
model = BigramLanguageModel()
m = model.to(device)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y =get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
optimizer = torch.optim.Adam(m.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses =estimate_loss()
        print(f"step:{iter}  train_loss{losses['train']}, val_loss {losses['val']}")
    xb,yb = get_batch("train")
    
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context = torch.zeros([1,1],dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))    