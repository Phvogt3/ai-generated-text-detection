# AI-Generated Text Detection

Do NLP models that detect AI written text actually learn *writing style*, or are they just picking up on
topic bias in the training data?

## Research question

> What detects AI generated text more effectively: word and topic content through TF-IDF, writing style
> patterns, or the two combined? And does a detector that scores 99% on a random split still work when it is
> tested on a subject it never trained on?

The second half of that question is what the project is built around. A detector that works by recognizing
which subjects AI essays tend to cover will collapse the moment the topic distribution shifts, so the only
way to tell the two apart is to take a topic away from the model and score it on that topic alone.

## Data

A [Kaggle data set](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) of roughly 500,000 essays
labeled human or AI generated, about 63% human and 37% AI. The comparison sections use a stratified 120,000
essay sample so they finish quickly. The streaming section trains on all 487,235 essays.

The file is not redistributed here. Download `AI_Human.csv` from the link above and point `DATA_PATH` at it.

## What the notebook does

**Feature sets.** TF-IDF with English stopwords removed, 15,000 features and unigrams plus bigrams, against
seven stylometric features covering essay length, word length, sentence length, punctuation counts,
uppercase counts and lexical diversity, and then the two stacked together.

**Models.** Multinomial Naive Bayes, Logistic Regression, a Linear SVM and a small neural network, scored on
accuracy, macro F1, ROC-AUC and fit time so that the effect of the classifier can be separated from the
effect of the features.

**Topic holdout.** Essays are clustered into six topics with MiniBatchKMeans on TF-IDF, using no labels, so
the clusters group on subject matter alone. The AI share inside each cluster is reported first, which is the
topic bias itself. Each cluster is then held out in turn: TF-IDF is refit on the remaining topics only and
the model is scored on the unseen topic, against a random split of identical size as the control. Results
are read on ROC-AUC and balanced accuracy rather than raw accuracy, because a cluster that is almost
entirely one class can be scored well by a model that always guesses that class.

**High dimensional behavior.** Logistic Regression, a neural network and a Random Forest are timed and
scored on the same sparse 15,000 column matrix, the Random Forest is run again on TruncatedSVD components to
show what dimension reduction does for a tree model, and a t-SNE of the essay space is colored by class and
by topic cluster.

**Full data training.** A HashingVectorizer maps terms into a fixed 2^20 column space with no vocabulary
held in memory, and SGDClassifier learns through `partial_fit` on one shuffled chunk at a time, with test
accuracy recorded after every chunk. The same loop works unchanged on a file too large to load, by reading
it with `pd.read_csv(..., chunksize=...)`.

**Closing summary.** Every figure quoted in the conclusion is printed from the values the notebook computed,
so the written claims cannot drift from the output.

## Results

On a random split of the 120,000 essay sample, word content carries almost all of the signal.

| Model | Feature set | Accuracy |
| --- | --- | --- |
| Naive Bayes | TF-IDF | 0.964 |
| Naive Bayes | Stylometric | 0.627 |
| Naive Bayes | Combined | 0.964 |
| Logistic Regression | Combined | 0.990 |

Style features on their own land near the majority class rate, and adding them on top of TF-IDF changes
almost nothing. The terms carrying the most weight are subject words rather than anything structural, which
is the first sign that the classifier is separating topics.

The topic holdout is the result that matters, and the notebook prints it directly. When a subject is removed
from training, whatever ROC-AUC survives is the part of the score that is about how AI writes rather than
about what the two classes happen to write about.

## Repository layout

```
notebook.ipynb         analysis with outputs and figures
ai_text_detection.py   full pipeline as a script
report/writeup.pdf     written report
```

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

The notebook mounts Google Drive when it runs in Colab and otherwise reads `DATA_PATH` or a local
`AI_Human.csv`. Sample sizes, the number of topic clusters and the streaming chunk size are all constants in
the configuration cell, so lowering them is the way to make a full run faster. A full run takes roughly 30
to 45 minutes on Colab with the high RAM option.

## Where this goes next

1. Rebuild the data set so both classes cover an identical set of prompts, then rerun the comparison. Any
   accuracy that survives is style signal.
2. Expand the style features with readability scores, transition word density, passive voice rate and
   paragraph structure, then run the topic holdout on style features alone, where topic bias cannot help.
3. Test transfer to other kinds of writing such as emails, news and forum posts, where the topic mix looks
   nothing like student essays.
4. Compare outputs from different language models to see whether each leaves its own detectable fingerprint.

## License

MIT, see `LICENSE`.
