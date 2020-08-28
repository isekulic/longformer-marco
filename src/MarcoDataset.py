import os
import pandas as pd
import string
import torch

from torch.utils.data import Dataset
import pytorch_lightning as pl

class MarcoDataset(Dataset):
    """
    Dataset abstraction for MS MARCO document re-ranking. 
    """
    def __init__(self, data_dir, mode, tokenizer, max_seq_len=512, args=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # load queries
        self.queries = pd.read_csv(os.path.join(self.data_dir, args.queries),
                                   sep='\t', header=None, names=['qid', 'query_text'], index_col='qid')
        self.relations = pd.read_csv(os.path.join(self.data_dir, f'msmarco-doc{mode}-qrels.tsv'),
                                   sep=' ', header=None, names=['qid', '0', 'did', 'label'])
        self.top100 = pd.read_csv(os.path.join(self.data_dir, f'msmarco-doc{mode}-top100'),
                                   sep=' ', header=None, names=['qid', 'Q0', 'did', 'rank', 'score', 'run'])

        self.load_documents_mode = 'lookup'
        # load documents TOO BIG TO LOAD THEM!
        if self.load_documents_mode == 'memory':
            self.documents = pd.read_csv(os.path.join(self.data_dir, 'msmarco-docs.tsv'),
                       sep='\t', header=None,
                       names=['did', 'url', 'title', 'doc_text'], index_col='did',)
        elif self.load_documents_mode == 'lookup':
            self.doc_seek = pd.read_csv(os.path.join(self.data_dir,'msmarco-docs-lookup.tsv'),
                                        sep='\t', header=None,
                                        names=['did', 'trec_offset', 'tsv_offset'],
                                        index_col='did')

        # downsample the dataset so the positive:negative ratio is 1:10
        if mode == 'train':
            self.top100 = self.top100.sample(frac=0.1, random_state=42).append(
                                self.relations[['qid', 'did']], ignore_index=True)
            self.top100.drop_duplicates(keep='first', inplace=True)
            # shuffle the data so positives are ~ evenly distributed
            self.top100 = self.top100.sample(frac=1, random_state=42).reset_index(drop=True) 

        elif mode == 'dev' and args.use_10_percent_of_dev:
            # use 10% of the data for dev during training
            import numpy as np; np.random.seed(42)
            queries = self.top100['qid'].unique()
            queries = np.random.choice(queries, int(len(queries)/50), replace=False)
            print(len(queries))
            self.top100 = self.top100[self.top100['qid'].isin(queries)]
        print(f'{mode} set len:', len(self.top100))

    # needed for map-style torch Datasets
    def __len__(self):
        return len(self.top100)

    # needed for map-style torch Datasets
    def __getitem__(self, idx):
        x = self.top100.iloc[idx]
        query = self.queries.loc[x.qid].query_text
        if self.load_documents_mode == 'memory':
            document = self.documents.loc[x.did].doc_text # too slow too big
        elif self.load_documents_mode == 'lookup':
            doc_file = open('../data/msmarco-docs.tsv', 'r', encoding='utf-8')
            # doc_file = open(os.path.join(self.data_dir, 'msmarco-docs.tsv'), 'r')
            file_offset = self.doc_seek.loc[x.did].tsv_offset
            doc_file.seek(file_offset, 0)
            line = doc_file.readline()
            doc_file.close()
            splited = line.split('\t')
            # when using num_workers > 1 seek get's fucked up
            assert(splited[0] == x.did)
            document = ' '.join(splited[3:])

        label = 0 if self.relations.loc[(self.relations['qid'] == x.qid) & (self.relations['did'] == x.did)].empty else 1

        tensors = self.one_example_to_tensors(query, document, idx, label)
        return tensors

    # main method for encoding the example
    def one_example_to_tensors(self, query, document, idx, label):

        encoded = self.tokenizer.encode_plus(query, document,
                        add_special_tokens=True,
                        max_length=self.max_seq_len,
                        truncation='only_second',
                        truncation_strategy='only_second',
                        return_overflowing_tokens=False,
                        return_special_tokens_mask=False,
                        return_token_type_ids=True,
                        pad_to_max_length=True
                            )
        encoded['attention_mask'] = torch.tensor(encoded['attention_mask'])

        encoded['input_ids'] = torch.tensor(encoded['input_ids'])

        encoded.update({'label': torch.LongTensor([label]),
                        'idx': torch.tensor(idx)})
        return encoded

