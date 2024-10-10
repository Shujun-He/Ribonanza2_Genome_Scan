from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import polars as pl
#tokens='ACGU().BEHIMSX'

def load_bpp(filename,seq_length=177):
    matrix = [[0.0 for x in range(seq_length)] for y in range(seq_length)]
    #matrix=0
    # data processing
    for line in open(filename):
        line = line.strip()
        if line == "":
            break
        i,j,prob = line.split()
        matrix[int(j)-1][int(i)-1] = float(prob)
        matrix[int(i)-1][int(j)-1] = float(prob)

    matrix=np.array(matrix)

    #ap=np.array(matrix).sum(0)
    return matrix

class RNADataset(Dataset):
    def __init__(self,indices,data_dict,k=5,train=True,flip=False):

        self.indices=indices
        self.data_dict=data_dict
        self.k=k
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.tokens['P']=4
        self.train=train
        self.flip=flip

    def generate_src_mask(self,L1,L2,k):
        mask=np.ones((k,L2),dtype='int8')
        for i in range(k):
            mask[i,L1+i+1-k:]=0
        return mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        idx=self.indices[idx]

        sequence=[self.tokens[nt] for nt in self.data_dict['sequences'][idx]]
        sequence=np.array(sequence)


        seq_length=len(sequence)

        #labels are in the order 2A3, DMS
        labels=self.data_dict['labels'][idx][:seq_length]
        errors=self.data_dict['errors'][idx][:seq_length]


        loss_mask = (labels==labels)

        label_mask=labels!=labels

        labels[label_mask]=0
        errors[errors!=errors]=0

        labels=labels.clip(0,1)

        sequence=torch.tensor(sequence).long()
        labels=torch.tensor(labels).float()
        loss_mask=torch.tensor(loss_mask).bool()
        #mask=torch.tensor(self.src_masks[idx])
        mask=self.generate_src_mask(seq_length,seq_length,self.k)
        mask=torch.tensor(mask)
        errors=torch.tensor(errors).float()

        SN=torch.tensor(self.data_dict['SN'][idx]).float()

        #id=self.data_dict['sequence_ids'][idx]
        #bpp=load_bpp(f"../../bpp_files_v2.0.3/{id}.txt",len(sequence))
        #bpp=torch.tensor(bpp).float()

        attention_mask=self.data_dict['attention_mask']



        if (self.train and np.random.uniform()>0.5) or self.flip:
            sequence=sequence.flip(-1)
            #bpp=bpp.flip(-1).flip(-2)
            #attention_mask=attention_mask.flip(-1).flip(-2)
            #mask=mask.flip(-1)
            labels=labels.flip(-2)
            loss_mask=loss_mask.flip(-2)


        data={'sequence':sequence,
              "labels":labels,
              "mask":mask,
              "loss_mask":loss_mask,
              "errors":errors,
              "SN":SN,
              "attention_mask":attention_mask}
        return data

class TestRNAdataset(RNADataset):

    
    def __getitem__(self, idx):

        #id=self.ids[idx]

        #rows=self.df.loc[self.df['id']==id].reset_index(drop=True)
        #print()
        #idx=int(idx)
        #print(self.tokens)
        sequence=[self.tokens[nt] for nt in self.data_dict['sequences'][idx]]
        sequence=np.array(sequence)

        seq_length=len(sequence)
        sequence=torch.tensor(sequence).long()
        mask=self.generate_src_mask(seq_length,seq_length,self.k)
        mask=torch.tensor(mask)
        #errors=torch.tensor(errors).float()

        id=self.data_dict['sequence_ids'][idx]
        # bpp=load_bpp(f"../../bpp_files_v2.0.3/{id}.txt",len(sequence))
        # bpp=torch.tensor(bpp).float()
        attention_mask=self.data_dict['attention_mask']
        data={'sequence':sequence,
              "mask":mask,
              "attention_mask":attention_mask}

        return data
def replace_non_AUGC_chars(s):
    valid_chars = set('AUGC')
    return ''.join(c if c in valid_chars else 'N' for c in s)
class SlidingWindowTestRNAdataset(RNADataset):

    def __init__(self,sequence,window_size=100, stride=50):

            self.sequence=sequence
            self.window_size=window_size
            self.stride=stride
            self.tokens={nt:i for i,nt in enumerate('ACGU')}
            self.tokens['P']=4
            self.tokens['N']=4
            self.tokens['R']=4
            self.tokens['M']=4

            self.segments=len(self.sequence)//self.stride

    def __len__(self):
        return self.segments

    def __getitem__(self, idx):

        #id=self.ids[idx]

        #rows=self.df.loc[self.df['id']==id].reset_index(drop=True)
        #print()
        #idx=int(idx)
        #print(self.tokens)
        start=idx*self.stride
        end=idx*self.stride+self.window_size
        sequence=[self.tokens[nt] for nt in replace_non_AUGC_chars(self.sequence[start:end])]
        if len(sequence)<self.window_size:
            sequence += (self.window_size-len(sequence))*[4]
        sequence=np.array(sequence)

        seq_length=len(sequence)
        sequence=torch.tensor(sequence).long()

        data={'sequence':sequence,
              'start_position':start}

        return data



class Custom_Collate_Obj:


    def __call__(self,data):
        length=[]
        for i in range(len(data)):
            length.append(len(data[i]['sequence']))
        max_len=max(length)


        sequence=[]
        labels=[]
        masks=[]
        loss_masks=[]
        errors=[]
        SN=[]
        attention_mask=[]
        for i in range(len(data)):
            to_pad=max_len-length[i]

            #if to_pad>0:
            sequence.append(F.pad(data[i]['sequence'],(0,to_pad),value=4))
            #masks.append(data[i]['mask'])
            masks.append(F.pad(data[i]['mask'],(0,to_pad),value=0))
            loss_masks.append(F.pad(data[i]['loss_mask'],(0,0,0,to_pad),value=0))
            #print(data[i]['labels'].shape)
            labels.append(F.pad(data[i]['labels'],(0,0,0,to_pad),value=0))
            errors.append(F.pad(data[i]['errors'],(0,0,0,to_pad),value=0))
            SN.append(data[i]['SN'])
            attention_mask.append(data[i]['attention_mask'])
        sequence=torch.stack(sequence)
        labels=torch.stack(labels)#.permute(0,2,1)
        masks=torch.stack(masks)
        loss_masks=torch.stack(loss_masks)#.permute(0,2,1)
        errors=torch.stack(errors)#.permute(0,2,1)
        SN=torch.stack(SN)
        attention_mask=torch.stack(attention_mask)
        # print(sequence.shape)
        # print(labels.shape)
        # exit()

        length=torch.tensor(length)

        data={'sequence':sequence,
              "labels":labels,
              "masks":masks,
              "loss_masks":loss_masks,
              "errors":errors,
              "SN":SN,
              "attention_mask":attention_mask,
              "length":length}

        return data

class Custom_Collate_Obj_test(Custom_Collate_Obj):

    def __call__(self,data):
        length=[]
        for i in range(len(data)):
            length.append(len(data[i]['sequence']))
        max_len=max(length)
        sequence=[]
        masks=[]
        attention_mask=[]
        for i in range(len(data)):
            to_pad=max_len-length[i]
            sequence.append(F.pad(data[i]['sequence'],(0,to_pad),value=4))
            masks.append(F.pad(data[i]['mask'],(0,to_pad),value=0))
            attention_mask.append(data[i]['attention_mask'])
            #masks.append(F.pad(data[i]['mask'],(0,to_pad,0,0),value=0))
        sequence=torch.stack(sequence)
        masks=torch.stack(masks)
        length=torch.tensor(length)
        attention_mask=torch.stack(attention_mask)

        data={'sequence':sequence,
              "masks":masks,
              "attention_mask":attention_mask,
              "length":length}

        return data
