# Rerank Experiments
## Training
Download the [limrank-data]() repository for training. 
Following Rank1, we use the following command to train the reranker.

```bash
# first clone llamafactory and install it. 
# add modify the data/configs.json 
# in current dir
git clone https://github.com/hiyouga/LLaMA-Factory.git
mkdir data/
cp train_configs/dataset_info.json data/
wget -P data https://huggingface.co/datasets/songtingyu/limrank-data/resolve/main/train_limrank.json
# huggingface download
cd LLaMA-Factory
pip install -e . # This step is to install LLaMA-Factory 
cd ..


# In rerank-exps dir
llamafactory-cli train train_configs/limrank.yaml
llamafactory-cli export train_configs/export_limrank.yaml
```

## Evaluation
### Data Preparation
Download the LimRank-Run-Files repository (required for evaluation)
git lfs install # if you don't have it already
git clone https://huggingface.co/datasets/songtingyu/limrank-run-files

### Enviroment Setup
```bash
git submodule update --init --recursive
cd mteb & pip install -e mteb
```

```bash
bash eval_scripts/eval_all.sh ['songtingyu/LimRank', 'to_export/limrank'] # This will evaluate all datasets mentioned in the paper.  
```