# AI Generated Text Detection

Machine learning project that tests whether NLP models detect AI generated essays by learning real writing style differences or by picking up on topic bias in the data.

## Research Question

Which works better for detecting AI generated text: word and topic features from TF-IDF, writing style features, or both combined? And does the choice of model (Naive Bayes or Logistic Regression) change the results?

## Data

The [AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) dataset from Kaggle has about 487,000 essays labeled as human written or AI generated. A random sample of 120,000 essays was used, split 80/20 into training and test sets with stratification. The classes are imbalanced, with about 63% human and 37% AI. The dataset is not included in this repository.

## Features

**TF-IDF:** 15,000 unigram and bigram features with English stopwords removed. These capture what the essay is about.

**Style features:** seven features that capture how the essay is written.

| Feature | Description |
|---|---|
| num_words | Total word count |
| avg_word_length | Average characters per word |
| avg_sentence_length | Average words per sentence |
| num_exclamations | Exclamation mark count |
| num_questions | Question mark count |
| num_uppercase | Uppercase character count |
| lexical_diversity | Unique words divided by total words |

## Results

| Model | Features | Test Accuracy |
|---|---|---|
| Naive Bayes | TF-IDF only | 96.4% |
| Naive Bayes | Style only | 62.7% |
| Naive Bayes | Combined | 96.4% |
| Logistic Regression | Combined | 99.0% |

Style features added almost nothing on top of TF-IDF, and on their own they barely beat always guessing "human." Logistic Regression did better than Naive Bayes on the same features.

The words that most strongly predicted each class were mostly topic related. Top human terms included "student_name," "teacher_name," and "driveless cars," and top AI terms included "university education," "zoos," and "public health." This suggests the high accuracy comes mostly from the two classes covering different essay prompts rather than from real differences in writing style.

AI essays in the sample were shorter, used longer words, had slightly higher lexical diversity, and used fewer question marks than human essays.

## Future Work

Retrain on a topic balanced subset to see how much accuracy drops without the topic shortcut. Add richer style features like readability scores and transition word density. Test on other kinds of text such as emails or news articles, and compare output from different LLMs.

## Files

`AI_Generated_Text_Detection.ipynb` is the full notebook with outputs.

## Running

```
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

Download `AI_Human.csv` from Kaggle and update the file path in the first code cell. The notebook was built in Google Colab, so remove the Google Drive mount lines if you run it locally.
