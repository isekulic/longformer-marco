# <p align=center>`Longformer for MS MARCO document ranking task`</p>

**\*\*\*\*\* 28. August: first commit \*\*\*\*\***

The checkpoints and further instructions on running the code will be updated shortly.

## About

We employ [Longformer](https://github.com/allenai/longformer), a BERT-like model for long documents, on the [MS MARCO](https://microsoft.github.io/msmarco/) document re-ranking dataset. 

## Learning setting

Due to the computing limitations, the **hyperparameters were not optimised**. We default to the following hyperparameters:
```
--lr=3e-05
--max_seq_len=4096
--num_warmup_steps=2500
```

For each query, we randomly sample 10 negative documents from the top 100 documents retrieved in the initial retrieval step. 

## Results

|       |Dev|Test|
|-------|---|----|
|MRR@100|0.3366|0.307|


The work is done by [Ivan Sekulic](https://isekulic.github.io/) (Università della Svizzera italiana), [Amir Soleimani](https://asoleimanib.github.io) (University of Amsterdam), [Mohammad Aliannejadi](https://aliannejadi.com/) (University of Amsterdam), and Fabio Crestani (Università della Svizzera italiana).

