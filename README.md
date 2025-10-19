# IMA312-MiniProject
This repository broadly will cover all the files involved in ICT Course : IMA312

Team Members :

- Amruth Ayaan
- Mukund Rathi
- Ritishaa Anand

We have implemented a research paper and benchmarked it against current industry standard compression technique and we documented our entire experiment in $\text{report.pdf}$

The following concepts from the course materials were used;

$$\text{Self Information :} \,I(X) = -\log_2(p_i)$$
$$\text{Avg Self Info/Shannon Entropy :} \,H[X] =E[I(X)] = -\sum_{i\ge 1} p_i \log_2(p_i)$$
$$\text{Mutual Information :} \,I(X;Y)=\sum_{x\in X} \sum_{y\in Y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

The repository contains two python scripts, one running a non-penalized variation ($\text{nonpenalizedchowliu.py}$) of the compression technique and one with the penalized ($\text{penalizedchowliu.py}$). 

MDL (Minimum description length) introduces a penalty to account for model complexity (metadata storage), enhancing the practical applicability of the Chow-Liu algorithm.
MI (Mutual information) serves as the core metric for detecting and weighting dependencies in the original and penalized approaches, driving the tree construction process.

The non-penalized approach refers to the "vanilla" Chow-Liu tree, which selects the maximum spanning tree by maximizing the sum of empirical mutual informations $\hat{I}(X_i; X_j)$ over the edges, without considering metadata costs. This approximates the joint distribution but can lead to high storage overhead for sparse pairwise histograms.

The penalized approach is the paper's improvement, using an MDL-like criterion to account for model description costs. It selects the tree $T^*$ that maximizes $\sum_{(i \to j) \in E} n \cdot \hat{I}(X_i; X_j) - \sum_{(i \to j) \in E} |cn(\hat{p}_{i,j})|$, where $n$ is the number of rows and $|cn(\hat{p}_{i,j})|$ is the encoding length of the empirical pairwise histogram. This penalizes edges with high metadata costs (as mentioned above for sparse tables), resulting in more efficient compression by balancing information gain against storage overhead.

To run this.

- Git clone the repo : git clone https://github.com/NotCleo/IMA312-ChowLiuCompression.git
- Create a virtual environment (if required) : python3 -m venv venv, source venv/bin/activate
- Install all the requirements : pip install -r requirements.txt
- Run the scripts : python3 penalizedchowliu.py, python3 nonpenalizedchowliu.py (we are lazy to make a bash script:p)





