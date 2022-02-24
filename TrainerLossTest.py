import torch
import numpy as np
import math


def train(model, iterator, optimizer, criterion,duration_loss,sig,clip, batch,device):
    model.train()
    epoch_loss = 0
    
    for src, trg in iterator:
       
        src=src.to(device)
        
        trg=trg.to(device)
    
        optimizer.zero_grad()
        
        output,length,KLD= model(src, trg[:,:-1])
       
        output_dim = output.shape[-1]
        
        loss1 = criterion(output.contiguous().view(-1, output_dim),trg[:,1:,:-1].contiguous().view(-1).long())
  
        loss2 = torch.exp(duration_loss(length.contiguous().view(-1).float(),trg[:,1:,-1:].contiguous().view(-1).float()))*100
        
        loss = (loss1 + loss2 +KLD) / batch
               
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/len(iterator) ,KLD ,loss1 ,loss2

def accuracy(iterator, model, device,sig, sos_idx , max_len =20):
    model.eval()
    pred=[]
    acc=0
    batch_accuracy=[]
    with torch.no_grad():
        for src,trg in iterator:
                src_tensor=src.to(device)
                trg=trg.to(device)
                src_mask = model.make_src_mask(src_tensor)
                enc_src = model.encoder(src_tensor, src_mask)
                trg_indexes = [[sos_idx,0]]
        
                
                for i in range(max_len):
                    
                    trg_tensor = torch.FloatTensor(trg_indexes).unsqueeze(0).to(device)
                    trg_mask = model.make_trg_mask(trg_tensor).to(device)
                    
                    output,length,attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
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
                
                examp_pred=[]
                for item in pred[1:]: 
                    duration=round(item[1]*100)
                    for i in range(duration):
                        if len(examp_pred)==len(examp_tar):
                            break
                        examp_pred.append(item[0])
                
                correct=0
                for x,y in zip(examp_pred,examp_tar):
                    leng=len(examp_tar)
                    if x==y:
                        correct += 1
                    acc=correct/leng
            
                batch_accuracy.append(acc)
            
    return sum(batch_accuracy)/len(batch_accuracy)

def accuracyMOC(iterator, model, device,sig, sos_idx,INPUT_DIM , max_len =20):
    model.eval()
    pred=[]
    acc=0
    batch_accuracy=[]
    with torch.no_grad():
        for src,trg in iterator:
                src_tensor=src.to(device)
                
                trg=trg.to(device)
               
                
                src_mask = model.make_src_mask(src_tensor)
                enc_src = model.encoder(src_tensor, src_mask)
                trg_indexes = [[sos_idx,0]]
        
                
                for i in range(max_len):
                    
                    trg_tensor = torch.FloatTensor(trg_indexes).unsqueeze(0).to(device)
                    trg_mask = model.make_trg_mask(trg_tensor).to(device)
                    output,length,attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
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
