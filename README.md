# "A priori" Data Value estimation

This code implements the research work described in paper: 

**"A Priori" Shapley Data Value Estimation**. A. Navia-Vázquez, J. Cid-Sueiro and M.A. Vázquez. Pattern Analysis and Applications, 2024. (Under review).   


## Content


**DV_Shapley.py** -> Main script

**Shapley.py**  -> Class that estimates the Shapley values given the list of permutations and their utilities

**Models.py** ->  Class that computes the model utilities for all combinations among workers

**LC_model** -> Logistic Classifier model class used in the experiments. 

**gen_datasets.py** -> script that generates the different test cases from datasets. **Note: To avoid legal issues, initial datasets are not provided and must be downloaded from their original websites, as indicated in the paper.**

This code supports the following utilities described in the paper:


## Utilities:


### **Naive**

* Separate statistics $g_s(S)$ in (4) and cosine similarity in (8). Tag "ss" in the code.

### **IOcorr** 

* $g_{rit}(S)$:  input-target correlation statistics in (6) and cosine similarity in (8). Tag "rxy_1" in the code.

### **Welch**: 

* utility (15) computed using statistics in (6). Tag "welch_rxy_t2dif" in the code.

### **Bhattacharyya**: 

* $BD-g_{it}$: utility (20) using $g_{it}$ in (5). Tag "BD_rxy_spheric" in the code.b

* $BD-g_{rit}$: isotropic approximation $g_{rit}$ in (6). Tag "BD_rxy_scalar" in the code.b

### **Gauss**: 

* $Gauss-g_c$: utility (27) using $g_c$ in (7). Tag "Gauss_spheric_paired" in the code.
* $Gauss-g_{rc}$: utility (27) using $g_{rc}$ in (7). Tag "Gauss_scalar_paired" in the code.


## Acknowledgement 

This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 824988. https://musketeer.eu/

![](./EU.png)