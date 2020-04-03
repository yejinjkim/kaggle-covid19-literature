from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
    
def inputid (inputsent,tokenizername):
    tokenizer = BertTokenizer.from_pretrained(tokenizername)
    input_ids = []
    for sent in tqdm(inputsent):
        sent= word_tokenize(sent)[0:500]
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        input_ids.append(encoded_sent)
    return input_ids

def maxwordnum(allsec):
    allsentlen=[]
    for i in tqdm(allsec):
        wordnu=len(i)
        allsentlen.append(wordnu)
    maxnum=max(np.array(allsentlen))
    return maxnum

def dxseqpadding (seq,maxnu):
    seq2=[]
    for i in tqdm(seq):
        stamp=len(i)
        i=np.pad(i,((0,maxnu-stamp)),'constant',constant_values=0)
        seq2.append(i)
    return seq2

def attid (inputsent):
    attention_masks = []
    for sent in tqdm(inputsent):
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks

def dataloader (trainval, test,args):
    train_inputs=trainval[0]
    train_inputs = torch.tensor(train_inputs)
    train_labels=trainval[1]
    train_labels = torch.tensor(train_labels)
    train_masks=trainval[2]
    train_masks = torch.tensor(train_masks)
    
    val_inputs=trainval[3]
    val_inputs = torch.tensor(val_inputs)
    val_labels=trainval[4]
    val_labels = torch.tensor(val_labels)
    val_masks=trainval[5]
    val_masks = torch.tensor(val_masks)
    
    test_inputs=test[0]
    test_inputs = torch.tensor(test_inputs)
    test_labels=test[1]
    test_labels = torch.tensor(test_labels)
    test_masks=test[2]
    test_masks = torch.tensor(test_masks)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)    
    train_dataloader = DataLoader(train_data, batch_size=args)
    
    validation_data = TensorDataset(val_inputs, val_masks, val_labels)    
    validation_dataloader = DataLoader(validation_data, batch_size=args)
        
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataloader = DataLoader(test_data, batch_size=args)
    
    return (train_dataloader,validation_dataloader,test_dataloader)
    
def bertrnn_process (args,trainsent,valsent,testsent,trainlabel,vallabel,testlabel):
    if args['science']==True:       
        trainsci=inputid(trainsent,args['modelname2'])
        valsci=inputid(valsent,args['modelname2'])
        testsci=inputid(testsent,args['modelname2'])
        trainnor=inputid(trainsent,args['modelname1'])
        valnor=inputid(valsent,args['modelname1'])
        testnor=inputid(testsent,args['modelname1'])
        maxnum=maxwordnum(testnor)
        trainsci=dxseqpadding(trainsci,maxnum)
        valsci=dxseqpadding(valsci,maxnum)
        testsci=dxseqpadding(testsci,maxnum)
        trainnor=dxseqpadding(trainnor,maxnum)
        valnor=dxseqpadding(valnor,maxnum)
        testnor=dxseqpadding(testnor,maxnum)
        trainsciatt=attid(trainsci)
        valsciatt=attid(valsci)
        testsciatt=attid(testsci)
        trainnoratt=attid(trainnor)
        valnoratt=attid(valnor)
        testnoratt=attid(testnor)
        nortrainval=(trainnor,trainlabel,trainnoratt,valnor,vallabel,valnoratt)
        scitrainval=(trainsci,trainlabel,trainsciatt,valsci,vallabel,valsciatt)
        scitest=(testsci,testlabel,testsciatt)
        nortest=(testnor,testlabel,testnoratt)
        norloder=dataloader (nortrainval, nortest,int(args['batch_size']))
        sciloder=dataloader (scitrainval, scitest,int(args['batch_size']))
    else : 
            
        trainnor=inputid(trainsent,args['modelname1'])
        valnor=inputid(valsent,args['modelname1'])
        testnor=inputid(testsent,args['modelname1'])
        maxnum=maxwordnum(testnor)
        trainnor=dxseqpadding(trainnor,maxnum)
        valnor=dxseqpadding(valnor,maxnum)
        testnor=dxseqpadding(testnor,maxnum)
        trainnoratt=attid(trainnor)
        valnoratt=attid(valnor)
        testnoratt=attid(testnor)
        nortrainval=(trainnor,trainlabel,trainnoratt,valnor,vallabel,valnoratt)        
        nortest=(testnor,testlabel,testnoratt)
        norloder=dataloader (nortrainval, nortest,int(args['batch_size']))       
        sciloder=[]
    return norloder,sciloder

