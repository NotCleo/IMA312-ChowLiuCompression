# IMA312-MiniProject
This repository broadly will cover all the files involved in ICT IMA312-MiniProject


Structured Learning with Undirected Tree Models 


We have $k$ random variables $X_1, X_2, \ldots, X_k$, which form the nodes of an undirected tree model.  What tree $T$ do we use?


To pick the best tree,

$$\hat{T}=\underset{T}{\text{argmax}}\{\underset{\theta_T}{\text{max}}\log \prod_{i=1}^n(P(x_1^{(i)}, x_2^{(i)}, \ldots, x_k^{(i)});T;\theta_T)\}$$

For specific tree $T$:

We need to simplify the inner argument, 

$$\max_{\theta_T} \log \prod_{i=1}^{n} p(x_1, \ldots, x_k^{(i)} ; T, \theta_T)$$

Given the tree structure $T$ (which pairs of variables are connected), find the parameter values $\theta_T$ (the probabilities or conditional probabilities for each node/edge) that make the observed data most likely.


$$\max_{\theta_T} \log \prod_{i=1}^{n} p(x_1^{(i)}, \ldots, x_k^{(i)} ; T, \theta_T)$$

Let's simplify the inner expression,

In a tree-structured model with a chosen root $r$, the joint probability can be decomposed as:

$$\log \prod_{i=1}^{n} \left[ p(x_r^{(i)}) \prod_{j \neq r} p(x_j^{(i)} | x_{\pi(j)}^{(i)}) \right]$$

Using property of logarithms, 
$$\sum_{i=1}^{n} \log p(x_r^{(i)}) + \sum_{j \neq r} \sum_{i=1}^{n} \log p(x_j^{(i)} | x_{\pi(j)}^{(i)})$$

The maximum likelihood estimate (MLE) for any discrete variable is its empirical distribution:

$$\sum_{i=1}^{n} \log \hat{p}(x_r^{(i)}) + \sum_{j \neq r} \sum_{i=1}^{n} \log \hat{p}(x_j^{(i)} | x_{\pi(j)}^{(i)})$$


$$n \left[ \sum_{a} \hat{p}_{X_r}(a) \log \hat{p}_{X_r}(a) + \sum_{j \neq r} \sum_{a,b} \hat{p}_{X_j, X_{\pi(j)}}(a,b) \log \frac{\hat{p}_{X_j | X_{\pi(j)}}(a|b)}{\hat{p}_{X_{\pi(j)}}(b)} \right]$$

Multiply and Divide by $\hat{p}_{x_j}(a)$ in the second summand,

$$n \left[ \sum_{a} \hat{p}_{X_r}(a) \log \hat{p}_{X_r}(a) + \sum_{j \neq r} \sum_{a,b} \hat{p}_{X_j, X_{\pi(j)}}(a,b) \log \frac{\hat{p}_{X_j | X_{\pi(j)}}(a|b)\cdot \hat{p}_{x_j}(a)}{\hat{p}_{X_{\pi(j)}}(b)\cdot \hat{p}_{x_j}(a)} \right]$$

$$\text{Kullback–Leibler divergence : }\,\, D_{KL}(P || Q) =  \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

$$n \left[ \sum_{a} \hat{p}_{X_r}(a) \log \hat{p}_{X_r}(a) + \sum_{j \neq r} D(\hat{p}_{x_j,x_{\pi(j)}}||\hat{p}_{x_j}\hat{p}_{x_{\pi(j)}}) \right]$$


$$n \left[ -\sum_{j \in V} \underset{\text{stays same}}{H(\hat{p}_{X_j})} + \sum_{(i,j) \in E} \underset{\text{depends on T}}{\hat{I}(X_i ; X_j)} \right]$$

Therefore we arrive at;

$$\max_{\theta_T} \log \prod_{i=1}^{n} p(x_1^{(i)}, \ldots, x_k^{(i)} ; T, \theta_T)\implies n \left[ -\sum_{j \in V} H(\hat{p}_{X_j}) + \sum_{(i,j) \in E} \hat{I}(X_i ; X_j) \right]$$


$$ \therefore \hat{T}=\underset{T}{\text{argmax}}\left(n \left[ -\sum_{j \in V} H(\hat{p}_{X_j}) + \sum_{(i,j) \in E} \hat{I}(X_i ; X_j) \right]\right)$$


Start with edge-less graph and add edges to build an optimum tree $T^*$, and we do via Chow Liu Algorithm;

$$\text{1. Start with no edges, just the nodes layed out}\\\text{2. Calculate Mutual information for all (i,j)}\\\text{3. Sort them in descending order}\\\text{4. Start adding edges between most to least empirical mutual information holding node pairs}\\\text{5. Skip pairs to avoid making a cycle}$$

---

$$\text{Self Information :} \,I(X) = -\log_2(p_i)$$
$$\text{Avg Self Info/Shannon Entropy :} \,H[X] =E[I(X)] = -\sum_{i\ge 1} p_i \log_2(p_i)$$
$$\text{Mutual Information :} \,I(X;Y)=\sum_{x\in X} \sum_{y\in Y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

---
