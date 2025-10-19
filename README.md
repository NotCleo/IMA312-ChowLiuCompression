# IMA312-MiniProject
This repository broadly will cover all the files involved in ICT Course : IMA312

Team Members :

- Amruth Ayaan
- Mukund Rathi
- Ritishaa Anand

We have implemented a research paper and benchmarked it against current industry standard compression technique and we documented our entire experiment in $\text{report.txt}$

The following concepts from the course materials were used;

$$\text{Self Information :} \,I(X) = -\log_2(p_i)$$
$$\text{Avg Self Info/Shannon Entropy :} \,H[X] =E[I(X)] = -\sum_{i\ge 1} p_i \log_2(p_i)$$
$$\text{Mutual Information :} \,I(X;Y)=\sum_{x\in X} \sum_{y\in Y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

The repository contains two python scripts, one running a non-penalized variation of the compression technique and one with the penalized. 



