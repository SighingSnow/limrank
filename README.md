<h1 align="center">LimRank: Less is More for Reasoning-Intensive Information Reranking</h1>

<h4 align="center">
    <p>
        <a href="#links">Model/Data Links</a> |
        <a href="#experiments">Experiments</a> |
        <a href="#usage">Usage</a> |
        <a href="#citing">Citation</a>
    <p>
</h4>

Official repository for paper [LimRank: Less is More for Reasoning-Intensive Information Reranking](https://arxiv.org/abs/2510.23544). 

## Links
| Resource | Description |
|:---------|:------------|
| [songtingyu/limrank-7b](https://huggingface.co/songtingyu/limrank) | The trained LimRank model based on Qwen2.5-7B |
| [songtingyu/limrank-data](https://huggingface.co/datasets/songtingyu/limrank-data) | The training datasets for limrank-7b  |
| [sogntingyu/limrank-results](https://huggingface.co/datasets/songtingyu/limrank-results) | The evaluation results of limrank-7b |
| [songtingyu/limrank-run-files](https://huggingface.co/datasets/songtingyu/limrank-run-files) | The running files to reproduce the results.  |


## Experiments
To reproduce the experiments, you can use the following code with uv for fast, reliable dependency management:

```bash
conda activate limrank_env
pip install -r requirements.txt
```

#### IR Experiments

Please refer to the [Rerank Experiments](./rerank-exps/README.md) for more details.

#### RAG Experiments

Please refer to the [LimRank-GPQA](./limrank-gpqa/README.md) for more details. 


## Citing
If you think our paper is useful, you can cite:

```bibtex
@misc{song2025limrankreasoningintensiveinformationreranking,
      title={LimRank: Less is More for Reasoning-Intensive Information Reranking}, 
      author={Tingyu Song and Yilun Zhao and Siyue Zhang and Chen Zhao and Arman Cohan},
      year={2025},
      eprint={2510.23544},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.23544}, 
}
```

## Acknowledgements
We would like to thank the authors of the following papers and repos for their open-source contributions. 
* [rank1](https://github.com/orionw/rank1)
* [ReasonIR](https://github.com/facebookresearch/ReasonIR) 
* [MTEB](https://github.com/embeddings-benchmark/mteb)

## License
[MIT](LICENSE)
