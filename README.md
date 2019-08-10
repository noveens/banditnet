# PyTorch implementation of BanditNet

## Dependencies
- torch >= 0.4.0
- tqdm
- numpy
- tensorboardX

## Instructions to run
1. `mkdir data/; cd data`
2. Download CIFAR-10 ([Link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)) in `data/`
3. `tar -xvf cifar-10-python.tar.gz`
4. `cd ..; python preprocess_cifar.py`
5. `cd code; python main.py`

## General Information
- Track all progress using tensorboard: 
  - `tensorboard --logdir code/tensorboard_stuff --port 16006`
  - `http://localhost:16006`
- `preprocess_cifar.py` makes bandit_dataset from the original cifar dataset
  - Edit the file to update hyper_parameters like `num_sample`
- All the hyper-parameters for the banditnet implementation can be set in the file `code/hyper_params.py`

## References
Paper: [https://www.cs.cornell.edu/people/tj/publications/joachims_etal_18a.pdf](https://www.cs.cornell.edu/people/tj/publications/joachims_etal_18a.pdf)

BibTeX: 
```
@InProceedings{Joachims/etal/18a,
  author = 	 {T. Joachims and A. Swaminathan and M. de Rijke},
  title = 	 {Deep Learning with Logged Bandit Feedback},
  booktitle =    {International Conference on Learning Representations (ICLR)},
  year = 	 2018}
```
