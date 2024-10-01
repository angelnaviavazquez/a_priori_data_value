# "A priori" Data Value estimation

## Content


**DV_Shapley.py** -> Main script

**Shapley.py**  -> Class that estimates the Shapley values given the list of permutations and their utilities

**Models.py** ->  Class that computes the model utilities for all combinations among workers

**LC_model** -> Logistic Classifier model class used in the experiments. 

**gen_datasets.py** -> script that generates the different test cases from datasets. **Note: To avoid legal issues, initial datasets are not provided and must be downloaded from their original websites, as indicated in the paper.**

This code supports the following utilities described in the paper:


## Utilities:


### **$Na¨ive$**

Separate statistics gs(S) in (4) and cosine similarity in (8)

### **$IO_{corr}$** 

* grit(S):  input-target correlation statistics  in (6) and cosine similarity in (8).

### Welch: 

* utility (15) (computed using statistics in (6)).

### Bhattacharyya: 

* utility (20) (using git in (5) (BD-git) 

* or its isotropic approximation grit in (6)) (BD-grit)

### Gauss: 

* Gauss-gc: utility (27) using gc in (7).
* Gauss-grc: utility (27) using grc in (7).


* Kmeans
* Neural networks
* Support Vector Machine
* Federated Budget Support Vector Machine
* Distributed Support Vector Machine


## Usage 




## Acknowledgement 

This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 824988. https://musketeer.eu/

![](./EU.png)