import math
import torch
import torch.nn as nn

import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import argparse

import time

import random

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
 
from BatchGen import BatchGenerator,LabelsDataset,randcuts
from Transformer import Seq2Seq,Encoder,Decoder,LIN,CVRAE_forecasting
from TrainerLossTest import train,accuracy,accuracyMOC



SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="50Salads")
parser.add_argument('--Epoch', default = '100',type=int)
parser.add_argument('--LR', default = '0.0005',type=float)
parser.add_argument('--Batch', default = '1',type=int)
parser.add_argument('--EncLayers', default = '3',type=int)
parser.add_argument('--DecLayers', default = '1',type=int)
parser.add_argument('--EncHead', default = '2',type=int)
parser.add_argument('--DecHead', default = '1',type=int)
parser.add_argument('--Sample', default = '20to50')
parser.add_argument('--device', default='0')
parser.add_argument('--mode', default='enc')
parser.add_argument('--split', default='1')


args = parser.parse_args()


device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

vid_list_file = "./"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./"+args.dataset+"/splits/test.split"+args.split+".bundle"

gt_path = "./"+args.dataset+"/groundTruth/"
gt_feat_path = "./"+args.dataset+"/features/"

arrays= "./Projects/VAEFeat/"+args.dataset+"/predictions/"


train_pth = "./VAEFeat/"+args.dataset+"/train"
test_pth = "./VAEFeat/"+args.dataset+"/test"

mapping_file = "./"+args.dataset+"/mapping.txt"

file_ptr = open(mapping_file, 'r') 
actions = file_ptr.read().split('\n')[:-1]
actions_dict=dict()
for a in actions:
    actions_dict[a.split()[1]] = (int(a.split()[0]))

SOS_index = actions_dict['SOS']
EOS_index = actions_dict['EOS']
    
batch = args.Batch

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    ## padd
    batch_input = [ t[0] for t in batch]
    batch_input = torch.nn.utils.rnn.pad_sequence(batch_input,batch_first=True)
    batch_targets = [ t[1] for t in batch]
    batch_targets = torch.nn.utils.rnn.pad_sequence(batch_targets,batch_first=True)
    return batch_input,batch_targets   


Batch_gen =  BatchGenerator(actions_dict,gt_path,vid_list_file,vid_list_file_tst,arrays,gt_feat_path,SOS_index,EOS_index)
input20,tar2010,tar2020,tar2030,tar2050,input30,tar3010,tar3020,tar3030,tar3050 = Batch_gen.GT_train_data_feat()  
input20_tst,tar2010_tst,tar2020_tst,tar2030_tst,tar2050_tst,input30_tst,tar3010_tst,tar3020_tst,tar3030_tst,tar3050_tst=Batch_gen.GT_test_data_feat()    
#input20_pred,tar2010_pred,tar2020_pred,tar2030_pred,tar2050_pred,input30_pred,tar3010_pred,tar3020_pred,tar3030_pred,tar3050_pred=Batch_gen.ASRF_pred()

#seq,tars=randcuts(args.dataset)

#trainin=(input20*4)
#traintar=tar2010+tar2020+tar2030+tar2050
#train_dataset = LabelsDataset(trainin,traintar)

#rand_dataset=LabelsDataset(seq,tars)

train_20to10_dataset = LabelsDataset(input20,tar2010)
train_20to20_dataset = LabelsDataset(input20,tar2020)
train_20to30_dataset = LabelsDataset(input20,tar2030)
train_20to50_dataset = LabelsDataset(input20,tar2050)

train_30to10_dataset = LabelsDataset(input30,tar3010)
train_30to20_dataset = LabelsDataset(input30,tar3020)
train_30to30_dataset = LabelsDataset(input30,tar3030)
train_30to50_dataset = LabelsDataset(input30,tar3050)

test_20to10_dataset = LabelsDataset(input20_tst,tar2010_tst)
test_20to20_dataset = LabelsDataset(input20_tst,tar2020_tst)
test_20to30_dataset = LabelsDataset(input20_tst,tar2030_tst)
test_20to50_dataset = LabelsDataset(input20_tst,tar2050_tst)

test_30to10_dataset = LabelsDataset(input30_tst,tar3010_tst)
test_30to20_dataset = LabelsDataset(input30_tst,tar3020_tst)
test_30to30_dataset = LabelsDataset(input30_tst,tar3030_tst)
test_30to50_dataset = LabelsDataset(input30_tst,tar3050_tst)

#pred_20to10_dataset = LabelsDataset(input20_pred,tar2010_pred)
#pred_20to20_dataset = LabelsDataset(input20_pred,tar2020_pred)
#pred_20to30_dataset = LabelsDataset(input20_pred,tar2030_pred)
#pred_20to50_dataset = LabelsDataset(input20_pred,tar2050_pred)

#pred_30to10_dataset = LabelsDataset(input30_pred,tar3010_pred)
#pred_30to20_dataset = LabelsDataset(input30_pred,tar3020_pred)
#pred_30to30_dataset = LabelsDataset(input30_pred,tar3030_pred)
#pred_30to50_dataset = LabelsDataset(input30_pred,tar3050_pred)

#rand_loader=DataLoader(rand_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

#train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

#train_20to10_loader = DataLoader(train_20to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#train_20to20_loader = DataLoader(train_20to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#train_20to30_loader = DataLoader(train_20to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#train_20to50_loader = DataLoader(train_20to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

#train_30to10_loader = DataLoader(train_30to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#train_30to20_loader = DataLoader(train_30to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#train_30to30_loader = DataLoader(train_30to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#train_30to50_loader = DataLoader(train_30to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

#test_20to10_loader = DataLoader(test_20to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#test_20to20_loader = DataLoader(test_20to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#test_20to30_loader = DataLoader(test_20to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#test_20to50_loader = DataLoader(test_20to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

#test_30to10_loader = DataLoader(test_30to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#test_30to20_loader = DataLoader(test_30to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#test_30to30_loader = DataLoader(test_30to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#test_30to50_loader = DataLoader(test_30to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

acc_20to10_loader = DataLoader(test_20to10_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_20to20_loader = DataLoader(test_20to20_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_20to30_loader = DataLoader(test_20to30_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_20to50_loader = DataLoader(test_20to50_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)

acc_30to10_loader = DataLoader(test_30to10_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_30to20_loader = DataLoader(test_30to20_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_30to30_loader = DataLoader(test_30to30_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_30to50_loader = DataLoader(test_30to50_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)

if args.Sample == 'random':
    train_loader = DataLoader(train_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_20to50_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '20to10':
    train_loader=DataLoader(train_20to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_20to10_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '20to20':
    train_loader=DataLoader(train_20to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_20to20_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '20to30':
    train_loader=DataLoader(train_20to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_20to30_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '20to50':
    train_loader=DataLoader(train_20to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_20to50_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '30to10':
    train_loader=DataLoader(train_30to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_30to10_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '30to20':
    train_loader=DataLoader(train_30to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_30to20_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '30to30':
    train_loader=DataLoader(train_30to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_30to30_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
elif args.Sample == '30to50':
    train_loader=DataLoader(train_30to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
    acc_loader = DataLoader(test_30to50_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
    




#pred_20to10_loader = DataLoader(pred_20to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#pred_20to20_loader = DataLoader(pred_20to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#pred_20to30_loader = DataLoader(pred_20to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#pred_20to50_loader = DataLoader(pred_20to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

#pred_30to10_loader = DataLoader(pred_30to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#pred_30to20_loader = DataLoader(pred_30to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#pred_30to30_loader = DataLoader(pred_30to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
#pred_30to50_loader = DataLoader(pred_30to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)



INPUT_DIM = len(actions_dict)+1
OUTPUT_DIM = INPUT_DIM
HID_DIM = 128
ENC_LAYERS = args.EncLayers
DEC_LAYERS = args.DecLayers
ENC_HEADS = args.EncHead
DEC_HEADS =  args.DecHead
ENC_PF_DIM = 128
DEC_PF_DIM = 128
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

lin = LIN(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)
if args.mode == 'linear':
    model = Seq2Seq(lin, dec, 0, 0, device)
else:
    model=CVRAE_forecasting(INPUT_DIM,128,128,2,device)
    #model = Seq2Seq(enc, dec, 0, 0, device)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f' Dataset: {args.dataset}\n Split: {args.split}\n Hidden Dimension: {HID_DIM}\n Encoder Layers: {ENC_LAYERS}\n \
Decoder Layers: {DEC_LAYERS}\n Encoder Heads: {ENC_HEADS}\n Decoder Heads: {DEC_HEADS}\n \
Learning Rate: {args.LR} \n Training Sample: {args.Sample}')
print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
#model.apply(initialize_weights);

LEARNING_RATE = args.LR
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
duration_loss=nn.MSELoss(reduction='sum')
sig=nn.Sigmoid()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = args.Epoch
data = args.Sample
CLIP = 1

Epoch_tacc=[]
Epoch_tloss=[]
Epoch_vloss=[]
Epoch_tacc=[]
Epoch_test=[]
Epoch_MSE=[]
Epoch_CE=[]
ACC_max=0
KL=[]

if args.action=="train":

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss,KLD,CE,MSE= train(model, train_loader, optimizer, criterion, duration_loss ,sig, CLIP,batch ,device)
        Epoch_CE.append(CE.item())
        Epoch_MSE.append(MSE.item())
        Epoch_tloss.append(train_loss)
        KL.append(KLD.item())
        
        train_MOC = model.Generate(train_loader,model,device,sig,SOS_index,INPUT_DIM)
        Epoch_tacc.append(np.mean(train_MOC))
        
        MOC = model.Generate(acc_loader,model,device,sig,SOS_index,INPUT_DIM)
        #MOC = model.Generate(acc_loader,model,device,sig,SOS_index,INPUT_DIM)
        Epoch_test.append(np.mean(MOC))
        
        print(np.mean(MOC))
        
        if np.mean(MOC)> ACC_max :
            ACC_max= np.mean(MOC)
            PATH="./MODELS/"+args.dataset+"/split"+str(args.split)+".pt"
            torch.save(model.state_dict(), PATH)
            print(f'Best model acc:{np.mean(MOC)}, saved at {PATH}' )
  
        end_time = time.time() 

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:2} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}') 
        
    epochs = range(1,101)
    plot1 = plt.figure(1)
    plt.plot(epochs,Epoch_CE, 'g', label='CE')
    plt.plot(epochs, Epoch_MSE, 'b', label='MSE')
    plt.plot(epochs,Epoch_tloss, 'r', label='Loss')
    plt.plot(epochs,KL, 'y', label='KLD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plot2 = plt.figure(2)  
    plt.plot(epochs, Epoch_tacc, 'r', label='train acc')
    plt.plot(epochs, Epoch_test, 'b', label='test acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    
    #plt.savefig("./MODELS/"+args.dataset+"/split"+str(args.split)+".png")

      
elif args.action == "test":
    PATH="./MODELS/"+args.dataset+"/split"+str(args.split)+".pt"

        
    model.load_state_dict(torch.load(PATH))
    print(f'model loaded{PATH}')
    print(np.mean(model.Generate(acc_20to10_loader,model,device,sig,SOS_index,INPUT_DIM)))
    print(np.mean(model.Generate(acc_20to20_loader,model,device,sig,SOS_index,INPUT_DIM)))
    print(np.mean(model.Generate(acc_20to30_loader,model,device,sig,SOS_index,INPUT_DIM)))
    print(np.mean(model.Generate(acc_20to50_loader,model,device,sig,SOS_index,INPUT_DIM)))
    print(np.mean(model.Generate(acc_30to10_loader,model,device,sig,SOS_index,INPUT_DIM)))
    print(np.mean(model.Generate(acc_30to20_loader,model,device,sig,SOS_index,INPUT_DIM)))
    print(np.mean(model.Generate(acc_30to30_loader,model,device,sig,SOS_index,INPUT_DIM)))
    print(np.mean(model.Generate(acc_30to50_loader,model,device,sig,SOS_index,INPUT_DIM)))
    

    


