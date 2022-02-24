import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import io
import random

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 2000):
        super().__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        #self.pos_embedding = PositionalEncoding(hid_dim, dropout,max_length)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.linear=nn.Linear(2048,hid_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        #one_hot= F.one_hot(src[:,:,:-1].squeeze(dim=-1).long(),num_classes=self.input_dim)
        #action_duration = src[:,:,-1:]
        #one_hotwlen=torch.cat((one_hot,action_duration),dim=2)
        
        embedding=self.linear(src) 
        
        src = self.dropout((embedding * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, mask = None)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
    
class LIN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 30):
        super().__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear=nn.Linear(input_dim+1,hid_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        #pos = [batch size, src len]
        one_hot= F.one_hot(src[:,:,:-1].squeeze(dim=-1).long(),num_classes=self.input_dim)
        action_duration = src[:,:,-1:]
        one_hotwlen=torch.cat((one_hot,action_duration),dim=2)
        embedding=self.linear(one_hotwlen) 
        
        #src = [batch size, src len, hid dim
        #src = [batch size, src len, hid dim]
            
        return embedding

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query) 
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
       
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
           
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
       
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
      
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 20):
        super().__init__()
        
        self.device = device
        self.output_dim=output_dim
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.linear=nn.Linear(output_dim+1,hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.fc_len=nn.Linear(hid_dim,1)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.sig= nn.Sigmoid()
        
        
    def forward(self, trg, enc_src,dec_t, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, trg len]

        trgone_hot= F.one_hot(trg[:,:,:-1].squeeze(dim=-1).long(),num_classes=self.output_dim)
        
        trgaction_duration = trg[:,:,-1:].float()
        
        trgone_hotwlen=torch.cat((trgone_hot,trgaction_duration),dim=2)
        
        trgembedding = self.linear(trgone_hotwlen) 
        
        trg = self.dropout((trgembedding * self.scale) + self.pos_embedding(pos))
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, dec_t,trg_mask, src_mask=None)
       
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        
        _length = self.fc_len(trg)
        length = self.sig(_length)
        #output = [batch size, trg len, output dim]
            
        return output,length,attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer2(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src,dec_t, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        
        _trg, attention = self.encoder_attention(trg ,dec_t, enc_src, enc_src, mask=None)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
    
    
class MultiHeadAttentionLayer2(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.device = device
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, sampled, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query) 
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        K=torch.cat((sampled,K),1)
        V=torch.cat((sampled,V),1)
        
        reshape = nn.Linear(K.shape[1],key.shape[1],device=self.device)
        
        K=reshape(K.permute(0,2,1)).permute(0,2,1)
        V=reshape(V.permute(0,2,1)).permute(0,2,1)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
       
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
           
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
       
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
      
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention    
    
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src[:,:,:-1].squeeze(dim=-1) != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg[:,:,:-1].squeeze(dim=-1) != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
    
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output,length,attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output,length
    
class CVRAE_forecasting(nn.Module):
    def __init__(self, act_dim, h_dim, z_dim, n_layers,device):
        super().__init__()

        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        self.device= device
        self.src_pad_idx = 0
        self.trg_pad_idx = 0

        self.x_enc = nn.Linear(2048, h_dim)
        self.act_enc =  nn.Linear(act_dim, h_dim)
        self.hidden_to_z = nn.Linear(h_dim, z_dim)
        self.phi_x_hidden_to_enc = nn.Linear(h_dim + h_dim, h_dim)
        self.hidden_to_prior = nn.Linear(h_dim, h_dim)
        self.z_to_phi_z = nn.Linear(z_dim, h_dim)
        self.phi_z_hidden_to_dec = nn.Linear(h_dim + h_dim, h_dim)
        self.dec_to_act = nn.Linear(h_dim, act_dim)
        self.dur_decoder = nn.Linear(h_dim + h_dim, 1)
        
        self.trans_enc = Encoder(self.act_dim,128,1,2,128,0.1,device)
        self.trans_dec = Decoder(self.act_dim,128,2,1,128,0.1,device)
        
        self.linear=nn.Linear(2048,h_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.gaussian_nll = nn.GaussianNLLLoss(full=True)

    def forward(self, act, trg):
        kld = 0
        
        phi_x_t = self.relu(self.x_enc(act))
    
        src_mask = self.make_src_mask(act)
        
        attention = self.trans_enc(act,src_mask)
        
        enc_t = self.relu(self.phi_x_hidden_to_enc(torch.cat([phi_x_t, attention], -1)))
        
        enc_mean_t = self.hidden_to_z(enc_t)
        
        enc_std_t = self.softplus(self.hidden_to_z(enc_t))

        unorm_prior_t=self.hidden_to_prior(attention)
        
        prior_t = self.relu(unorm_prior_t) #pass hidden state thorugh linear then relu
        
        prior_mean_t = self.hidden_to_z(prior_t) #hidden state mean
        
        prior_std_t = self.softplus(self.hidden_to_z(prior_t)) #hidden state std
       
        kld += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)  #calculate KL Divergence

        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
        #sampled using std
        
        phi_z_t = self.relu(self.z_to_phi_z(z_t))
        #passing sample to linear layer from z_dim to h_dim

        trg_mask = self.make_trg_mask(trg)
        
        dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([phi_z_t, attention], -1)))

        output,length,attention = self.trans_dec(trg,attention,dec_t,trg_mask, src_mask)
        #passing through linear layer to get to action probabilities
        
        return output,length,kld

    def init_hidden(self, obs_seq):
        return torch.randn(self.n_layers, obs_seq.size(1), self.h_dim)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.randn(std.shape, requires_grad=True, device=self.device)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return .5 * torch.sum(kld_element)
    
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src[:,:,:-1].squeeze(dim=-1) != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg[:,:,:-1].squeeze(dim=-1) != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
    
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask
    
    def Generate(self,iterator, model, device,sig, sos_idx,INPUT_DIM ,max_len =20):
        model.eval()
        pred=[]
        acc=0
        batch_accuracy=[]
        with torch.no_grad():
            for src,trg in iterator:
                    src_tensor=src.to(device)
                    trg=trg.to(device)
                    src_mask =self.make_src_mask(src_tensor)
                    
                    attention = self.trans_enc(src_tensor, src_mask)
                    
                    #embedding=self.linear(src) 
                    
                    phi_x_t = self.relu(self.x_enc(src_tensor))
                    
                    enc_t = self.relu(self.phi_x_hidden_to_enc(torch.cat([phi_x_t, attention], -1)))
                    enc_mean_t = self.hidden_to_z(enc_t)
                    enc_std_t = self.softplus(self.hidden_to_z(enc_t))
                    
                    z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                    phi_z_t = self.relu(self.z_to_phi_z(z_t))
                    
                    trg_indexes = [[sos_idx,0]]
            
                    for i in range(max_len):
                        
                        trg_tensor = torch.FloatTensor(trg_indexes).unsqueeze(0).to(device)
                        trg_mask = model.make_trg_mask(trg_tensor).to(device)
                        
                        output,length,attention_ = self.trans_dec(trg_tensor, attention, phi_z_t, trg_mask, src_mask)
                        pred_tokens = output.argmax(dim=2)[:,-1].item()
                        pred_len = length[:,-1].item()
                        trg_indexes.append([pred_tokens,pred_len])
                     
                    tar=trg.squeeze().cpu().detach().numpy()
                    examp_tar=[]
                    for item in tar[1:-1]: 
                        duration=round(item[1]*100)
                        for i in range(duration):
                            examp_tar.append(int(item[0]))
                   
                    
                    pred=trg_indexes
                    
                    n_T=np.zeros(INPUT_DIM)
                    n_F=np.zeros(INPUT_DIM)
                    
                    examp_pred=[]
                    for item in pred[1:]: 
                        duration=math.ceil(item[1]*100)
                        for i in range(duration):
                            if len(examp_pred)==len(examp_tar):
                                break
                            else:
                                examp_pred.append(item[0])
                                
                    while len(examp_pred)!=len(examp_tar):
                         examp_pred.append(0)
                            
                    for i in range(len(examp_tar)):
                        if examp_tar[i]==examp_pred[i]:
                            n_T[examp_tar[i]]+=1
                        else:
                            n_F[examp_tar[i]]+=1    
                   
                    for i in range(INPUT_DIM):
                        if n_T[i]+n_F[i] !=0:
                            batch_accuracy.append(float(n_T[i])/(n_T[i]+n_F[i]))
                    
        return batch_accuracy
    
    def accuracyMOC(self,iterator, model, device,sig, sos_idx,INPUT_DIM , max_len =20):
        model.eval()
        pred=[]
        acc=0
        batch_accuracy=[]
        with torch.no_grad():
            for src,trg in iterator:
                    src_tensor=src.to(device)
                    trg=trg.to(device)
                    src_mask = self.make_src_mask(src_tensor)
                    enc_src = self.trans_enc(src_tensor, src_mask)
                    trg_indexes = [[sos_idx,0]]
            
                    
                    for i in range(max_len):
                        
                        trg_tensor = torch.FloatTensor(trg_indexes).unsqueeze(0).to(device)
                        trg_mask = self.make_trg_mask(trg_tensor).to(device)
                        
                        output,length,attention = self.trans_dec(trg_tensor, enc_src, trg_mask, src_mask)
                        pred_tokens = output.argmax(dim=2)[:,-1].item()
                        pred_len = length[:,-1].item()
                        trg_indexes.append([pred_tokens,pred_len])
                     
                    tar=trg.squeeze().cpu().detach().numpy()
                    examp_tar=[]
                    for item in tar[1:-1]: 
                        duration=round(item[1]*100)
                        for i in range(duration):
                            examp_tar.append(int(item[0]))
                   
                    
                    pred=trg_indexes
                    
                    n_T=np.zeros(INPUT_DIM)
                    n_F=np.zeros(INPUT_DIM)
                    
                    examp_pred=[]
                    for item in pred[1:]: 
                        duration=math.ceil(item[1]*100)
                        for i in range(duration):
                            if len(examp_pred)==len(examp_tar):
                                break
                            else:
                                examp_pred.append(item[0])
                                
                    while len(examp_pred)!=len(examp_tar):
                         examp_pred.append(0)
                            
                    for i in range(len(examp_tar)):
                        if examp_tar[i]==examp_pred[i]:
                            n_T[examp_tar[i]]+=1
                        else:
                            n_F[examp_tar[i]]+=1    
                   
                    for i in range(INPUT_DIM):
                        if n_T[i]+n_F[i] !=0:
                            batch_accuracy.append(float(n_T[i])/(n_T[i]+n_F[i]))
                    
        return batch_accuracy

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)