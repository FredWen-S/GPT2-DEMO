import torch,torch.nn as nn,torch.nn.functional as F, math


class GPTconfig:
    def __init__(self,vocab_size=50257,block_size=256,n_layer=4,n_head=4,n_embd=128,dropout=0.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

        self.dropout = nn.Dropout(cfg.dropout)
        self.resid = nn.Dropout(cfg.dropout)

        self.register_buffer('mask',torch.tril(torch.ones(cfg.block_size,cfg.block_szie)).view(1,1,cfg.block_size,cfg.blovk_size))

    def forward(self,x):
        B,T,C= x.size()
        q,k,v=self.c_attn(x).split(self.n_embd,dim=2)

        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1, 2)
        q =q.view(B,T,self.n_head,C//self.n_head).transpose(1, 2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1, 2)


        att = (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))

        att=att.masked_fill(self.mask[:,:,:T,:T]==0,float('inf'))
        att=F.softmax(att,dim=-1)
        att =self.attn_dropout(att)
        y =att @ v

        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_dropout(self.c_proj(y))




