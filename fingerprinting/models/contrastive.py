
import torch
import torch.nn as nn
from fingerprinting.models.ntxent import NTXent

class ContrastiveFingerprint(nn.Module):
    
    def __init__(self,
                 encoder,
                 head_dims = [[512,128]],
                 temperature = 0.1,
                 feat_extract_head = -2,
                 **kwargs):
        super(ContrastiveFingerprint,self).__init__()
        
        self.encoder = encoder
        self.head_dims = head_dims
        self.encoder_dim = self.encoder.embed_dim
        self.heads = []
        
        for dim in head_dims:
            head = []
            last_dim = self.encoder_dim
            for d in dim:
                head.append(nn.Linear(last_dim,d,bias = False))
                head.append(nn.ReLU())
                last_dim = d
            self.heads.append(nn.Sequential(*head))
            
        self.heads = nn.ModuleList(self.heads)
        self.temperature = temperature
        self.loss = NTXent(temperature = temperature)
        self.feat_extract_head = feat_extract_head
        
        if self.feat_extract_head == -2:
            self.embed_dim = sum([dim[-1] for dim in self.head_dims])
        elif self.feat_extract_head == -1:
            self.embed_dim = self.encoder_dim
        elif self.feat_extract_head >= 0:
            self.embed_dim = self.head_dims[self.feat_extract_head]
        
        
    def forward(self,x):
        wav = x['audio']
        if wav.dim() == 4:
            wav = wav.contiguous().view(-1,1,wav.shape[-1]) ## [B*N_augmentations,T]
        elif wav.dim() == 5: # spectrogram : [B*N_augmentations,1,F,T]
            wav = wav.contiguous().view(-1,1,wav.shape[-2],wav.shape[-1])
                
        encoded = self.encoder(wav)
        projected = [head(encoded) for head in self.heads]
        
        return {
            'projected':projected,
            'encoded':encoded,
            "wav":wav,
        }
        
    
    def extract_features(self,x,head=None):
        
        # head -1 means the superspace above all heads
        # head -2 means the concatenated space of all heads
        # head n means the nth head
        if head is None:
            head = self.feat_extract_head

        with torch.no_grad():
            out_ = self({
                'audio':x,
            })
            
            if head == -1:
                return {'encoded': out_['encoded']}
            
            if head == -2:
                return {"encoded" : torch.cat(out_['projected'],dim=-1)}
            
            return {"encoded": out_['projected'][head]}
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()