import pandas as pd
import numpy as np
from torch.utils import data
import torch

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
MAX_SEQ_DRUG = 100
from sklearn.preprocessing import OneHotEncoder
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T   

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def encode_drug(df_data,  column_name = 'SMILES', save_column_name = 'drug_encoding'):
    unique = pd.Series(df_data[column_name].unique()).apply(trans_drug)
    unique_dict = dict(zip(df_data[column_name].unique(), unique))
    df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
    return df_data

def create_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    return train, val, test

def dataprocess(filename):
    df=pd.read_csv(filename)
    smiles=df['smiles'].values
    y=df['HIV_active'].values

    df_data = pd.DataFrame(zip(smiles, y))
    df_data.rename(columns={0:'SMILES',
								1: 'Label'}, 
								inplace=True)
    frac=[0.7,0.1,0.2]    
    df_data = encode_drug(df_data)
    train, val, test = create_fold(df_data, 123, frac)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class data_process_loader_Property_Prediction(data.Dataset):

	def __init__(self, list_IDs, labels, df):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.df = df
	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		index = self.list_IDs[index]
		v_d = self.df.iloc[index]['drug_encoding']        
		v_d = drug_2_embed(v_d)
		y = self.labels[index]
		#y=y.astype(np.float32)
		v_d=v_d.astype(np.float32)
		#y=y.type(torch.cuda.FloatTensor)
		#v_d=v_d.type(torch.cuda.FloatTensor)
		return v_d, y














