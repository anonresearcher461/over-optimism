# Over-optimistic evaluation and reporting of novel cluster algorithms: An illustrative study

When researchers publish new cluster algorithms, they usually demonstrate the strengths of their novel approaches by comparing the algorithms' performance with existing competitors. However, such studies are likely to be optimistically biased towards the new algorithms, as the authors have a vested interest in presenting their method as favorably as possible in order to increase their chances of getting published. 

Therefore, the superior performance of newly introduced cluster algorithms is over-optimistic and might not be confirmed in independent benchmark studies performed by neutral and unbiased authors. We present an illustrative study to illuminate the mechanisms by which authors - *consciously or unconsciously* - paint their algorithms' performance in an over-optimistic light.  

Using the recently published cluster algorithm Rock as an example, we demonstrate how optimization of the used data sets or data characteristics, of the algorithm's parameters and of the choice of the competing cluster algorithms leads to Rock's performance appearing better than it actually is. Our study is thus a cautionary tale that illuminates how easy it can be for researchers to claim apparent *'superiority'* of a new cluster algorithm. We also discuss possible solutions on how to avoid the problems of over-optimism.

## Reproduce

In order to exactly reproduce our results, you should create a virtual Python environment and install our requirements. 
First of all, if you do not have Python installed you can go to the official Python website and download Python version 3.9.5.
Next go to your version of our git in your command line and run the following command to create the virtual environment:

``` 
python -m venv your_environment_name 
```
Activate your virtual environment.  
*Windows*:
```
your_environment_name\Scripts\activate
```
*Mac OS / Linux*
```
source your_environment_name/bin/activate
```
Then install the requirements using our [requirements.txt](./requirements.txt)
(this ensures that you use exactly the same versions of the Python libraries as we did).
```
pip install -r requirements.txt
```
Now you can start the jupyter notebooks by calling
```
jupyter notebook
```
## Optimizing datasets and data characteristics

We examine how strongly choosing the "best" properties of a dataset can influence the performance estimation of Rock.

### Optimization of the data parameters with TPE

We used the optuna<sup>[[1]](#optuna)</sup> hyperparameter optimization framework in order to find the data parameter configurations for three popular synthetic datasets (Two Moons, Blobs and Rings) that yield the largest performance difference between Rock and the best of the competitors (which are not necessarily the data parameter configurations that yield the best **absolute** performance of Rock). The competing algorithms we chose to compare to Rock are DBSCAN, k-means, Mean-Shift and Spectral Clustering. The implementation of Rock we used can be found [here](./rock.py). For the competing methods we used the implementations from the sklearn.cluster<sup>[[2]](#cluster)</sup> module.

To simulate a researcher searching for the best data parameters, we performed the following formal optimization task using the TPE sampler<sup>[[3]](#sampler)</sup> from optuna:

<img src="https://render.githubusercontent.com/render/math?math=\text{argmax}_{D \in \mathcal{D}} \left\{ \frac{1}{10} \sum_{i = 1}^{10}  \Big( AMI\left(Rock(D^i), y_{D^i}\right) - \text{max}_{C \in \mathcal{C}}  AMI\left(C(D^i), y_{D^i}\right) \Big) \right\}">

For each dataset we created a jupyter notebook which you can find via the following links:
- [**Two Moons**](./notebooks/Optimizations/Overoptimism_Two_Moons.ipynb)
- [**Blobs** (with different densities)](./notebooks/Optimizations/Overoptimism_Den_Blobs.ipynb)
- [**Rings**](./notebooks/Optimizations/Overoptimism_Rings.ipynb)

Results for each of our runs can be found in the corresponding csv files and optuna study databases in the [**results/optimization folder**](./results/optimization). 
We provide a notebook that loads these results and creates figures used in the paper [**here**](./notebooks/Optimizations/Optuna_Results_Analysis.ipynb).

### Varying the data parameters 
After determining the optimal values for the data parameters, we analyzed the performance of Rock for non-optimal parameter values. That is, for each dataset and single data parameter in turn, the parameter was varied over a list of values, while the other data parameters were kept fixed at their optimal values. 

1. We kept the optimal jitter value and varied the [**number of samples**](./notebooks/Comparisons/Two_Moons_Analysis-num_samples.ipynb) for the Two Moons Dataset. 
2. We kept the optimal number of samples and varied the [**jitter**](./notebooks/Comparisons/Two_Moons_Analysis_jitter.ipynb) for the Two Moons Dataset. 
3. We kept the optimal number of samples and varied the [**number of dimensions**](./notebooks/Comparisons/Den_Blobs_Analysis.ipynb) for the Blobs dataset. 

Based on the results, the figures for the paper were generated with [**this notebook**](./notebooks/Comparisons/Generate_Comparison_Figures.ipynb). 

### Influence of the random seed

In the experiments given so far, we always considered the AMI averaged over ten random seeds. We now specifically study the influence of individual random seeds. We take the Two Moons dataset as an example, with a data parameter setting which is not optimal for Rock, but for which DBSCAN performs very well. We generate 100 datasets with these characteristics by setting 100 different random seeds, to check whether there exist particular seeds for which Rock does perform well, leading to over-optimization potential. This experiment can be found in [**this notebook**](./notebooks/Comparisons/Analysis_two_moons_100_seed.ipynb).

## Optimizing the algorithm's parameters

We varied [**Rock's hyperparameter t_max**](./notebooks/Optimizations/ROCK_Hyperparameter_Search.ipynb) (maximum number of iterations). Here we considered the **absolute** performance of Rock, given researchers would also strive to maximize the absolute performance of their novel algorithm. As exemplary datasets, we again consider Two Moons, Blobs and Rings, and additionally four real datasets frequently used for performance evaluation: Digits, Wine, Iris and Breast Cancer as provided by sci-kit<sup>[[4]](#optuna)</sup>. The data parameter settings for the three synthetic datasets (number of samples, amount of jitter etc.) correspond to the optimal settings from the optimizations above. We used a single random seed to generate the illustrative synthetic datasets.

In a next step, using the Two Moons dataset as an example, we compare Rock with DBSCAN with respect to the AMI performances over ten random seeds, first without, then with hyperparameter optimization (HPO). The HPO for Rock can be found [**here**](./notebooks/Optimizations/Two_Moons_ROCK_Hyperparameter_Search.ipynb) and the HPO for DBSCAN [**here**](./notebooks/Optimizations/Two_Moons_DBSCAN_Hyperparameter_Search.ipynb). We used the TPE for the HPO of DBSCAN (here, the TPE was not intended to model a researcher's behavior, but was used as a classical HPO method). The comparison illustrates the effect of neglecting parameter optimization for competing algorithms. 

---
<a name="optuna">[1]</a> https://optuna.org/  
<a name="cluster">[2]</a> https://scikit-learn.org/stable/modules/clustering.html  
<a name="sampler">[3]</a> https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html  
<a name="sampler">[4]</a> https://scikit-learn.org/stable/datasets/toy_dataset.html
