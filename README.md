# <p align=center>`Longformer for MS MARCO document ranking task`</p>

## About

We employ [Longformer](https://github.com/allenai/longformer), a BERT-like model for long documents, on the [MS MARCO](https://microsoft.github.io/msmarco/) document re-ranking dataset. More details about our model and experimental setting can be found in our [paper](https://arxiv.org/abs/2009.09392).

## Learning setting

Due to the computing limitations, the **hyperparameters were not optimised**. We default to the following hyperparameters:
```
--lr=3e-05
--max_seq_len=4096
--num_warmup_steps=2500
```

For each query, we randomly sample 10 negative documents from the top 100 documents retrieved in the initial retrieval step. 

## Training the model

To train the model, first download all of the necessary data, as described in [data/README.md](data/README.md). File names should match the filenames in [MarcoDataset.py](src/MarcoDataset.py).

You can then train with:
```
python run_longformer_marco.py
```
You can check all available hyperparameters with:
```
python run_longformer_marco.py --help
```

## Results

|       |Dev|Test|
|-------|---|----|
|MRR@100|0.3366|0.307|


The work is done by [Ivan Sekulic](https://isekulic.github.io/) (Università della Svizzera italiana), [Amir Soleimani](https://asoleimanib.github.io) (University of Amsterdam), [Mohammad Aliannejadi](https://aliannejadi.com/) (University of Amsterdam), and Fabio Crestani (Università della Svizzera italiana).

## Citing

Please consider citing our [paper](https://arxiv.org/abs/2009.09392) if you use our code or models:

    @misc{sekuli2020longformer,
    title={Longformer for MS MARCO Document Re-ranking Task},
    author={Ivan Sekulić and Amir Soleimani and Mohammad Aliannejadi and Fabio Crestani},
    year={2020},
    eprint={2009.09392},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
    }
