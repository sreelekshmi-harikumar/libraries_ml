
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.misc"]

tr = fetch_20newsgroups(subset="train", categories=cats)
te = fetch_20newsgroups(subset="test", categories=cats)

cv = CountVectorizer(stop_words="english")
Xtr, Xte = cv.fit_transform(tr.data), cv.transform(te.data)

for a in [1e-10, 0.01, 0.1, 1, 10]:
    m = MultinomialNB(alpha=a).fit(Xtr, tr.target)
    print(f"alpha={a:<6} | Accuracy: {accuracy_score(te.target, m.predict(Xte)):.4f}")
