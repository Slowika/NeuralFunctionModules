# Neural Function Modules

[![Python 3.6.8](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Official code for the paper [Neural Function Modules with Sparse Arguments: A Dynamic Approach to Integrating Information across Layers](https://arxiv.org/abs/2010.08012) ([Alex Lamb](https://sites.google.com/view/alexmlamb), [Anirudh Goyal](https://anirudh9119.github.io/), [Agnieszka Słowik](https://slowika.github.io/), [Philippe Beaudoin](https://twitter.com/PhilBeaudoin?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor), [Michael C. Mozer](https://home.cs.colorado.edu/~mozer/index.php), [Yoshua Bengio](https://yoshuabengio.org/)).

![alt text](https://github.com/Slowika/NeuralFunctionModules/blob/main/figures/nfm.png)

## Getting started

```
$ conda create -n nfm python=3.6.8
$ conda activate nfm
$ pip install -r requirements.txt
```

```
python sort_of_clevr_generator.py
python main.py --seed=5 --epochs=150
```
## Citation

```
@InProceedings{pmlr-v130-lamb21a,
  title = 	 { Neural Function Modules with Sparse Arguments: A Dynamic Approach to Integrating Information across Layers },
  author =       {Lamb, Alex and Goyal, Anirudh and S\l{}owik, Agnieszka and Mozer, Michael and Beaudoin, Philippe and Bengio, Yoshua},
  booktitle = 	 {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {919--927},
  year = 	 {2021},
  editor = 	 {Banerjee, Arindam and Fukumizu, Kenji},
  volume = 	 {130},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--15 Apr},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v130/lamb21a/lamb21a.pdf},
  url = 	 {
http://proceedings.mlr.press/v130/lamb21a.html
},
  abstract = 	 { Feed-forward neural networks consist of a sequence of layers, in which each layer performs some processing on the information from the previous layer. A downside to this approach is that each layer (or module, as multiple modules can operate in parallel) is tasked with processing the entire hidden state, rather than a particular part of the state which is most relevant for that module. Methods which only operate on a small number of input variables are an essential part of most programming languages, and they allow for improved modularity and code re-usability. Our proposed method, Neural Function Modules (NFM), aims to introduce the same structural capability into deep learning. Most of the work in the context of feed-forward networks combining top-down and bottom-up feedback is limited to classification problems. The key contribution of our work is to combine attention, sparsity, top-down and bottom-up feedback, in a flexible algorithm which, as we show, improves the results in standard classification, out-of-domain generalization, generative modeling, and learning representations in the context of reinforcement learning. }
}

```
