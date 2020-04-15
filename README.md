<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# InformationRetrieval
This demo implemented BM25 algorithm, pointwise model of logistic regression and pairwise model of RankNet and LambdaMART.

The RankNet is based on PyTorch and LambdaMART is based on XGBoost.
Pointwise model is implemented from scratch with logistic regression theory.



$$w_j := w_j -\alpha \frac{\partial J(w)}{w_j}$$
$$b := b-\alpha\frac{1}{m}\sum_{i=1}^m(p(w^T x^i)-y)$$
