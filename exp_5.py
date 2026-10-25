import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load and preprocess
rm = ("headers", "footers", "quotes")
tr = fetch_20newsgroups(subset="train", remove=rm)
te = fetch_20newsgroups(subset="test", remove=rm)
cv = CountVectorizer(stop_words="english", max_features=2000)
Xtr, Xte = cv.fit_transform(tr.data), cv.transform(te.data)

# 2. MLE: word probabilities per class = count / total
counts = np.array([Xtr[tr.target == k].sum(axis=0).A1 for k in range(20)])
V = counts.shape[1]
mle = counts / counts.sum(axis=1, keepdims=True)
print("MLE zero probabilities:", (mle == 0).mean())

# 3 & 4. MAP with Dirichlet priors, compared by accuracy
for a in [1e-10, 0.1, 1, 10, 100]:
    map_est = (counts + a) / (counts.sum(axis=1, keepdims=True) + a * V)
    m = MultinomialNB(alpha=a).fit(Xtr, tr.target)
    print(f"alpha={a:<7} | zero probs: {(map_est == 0).mean():.3f} | accuracy: {m.score(Xte, te.target):.4f}")
