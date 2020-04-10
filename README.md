# kaggle-covid19-literature

Requirements
```
pip install tqdm boto3 requests regex sentencepiece sacremoses
pip install transformers
```

![Overview](https://github.com/yejinjkim/kaggle-covid19-literature/blob/master/image%20(1).png)
1. Raw data
```
./raw-data # the literature files provided from Kaggle
```

2. Pseudo labelling with keywords and Snorkel (Kanglin, Shana, Qian)
```
./snorkel-pseudo-label/*keywords.csv* #keywords csv files for the 10 questions.

./snorkel-pseudo-label/*code.ipynb* #code for the pseudo label using Snorkel. 

./snorkel-pseudo-label/*pseudo-label.pkl* # estimated pseudo label for each question. pd.Dataframe with columns=['qid','sentence_sha', 'prob']
```
3. Retrieve relevant sentences (Kanglin, Yejin)

```
./sentence-classification/*code.ipynb* #code for sentence classification to retreive relevant sentences to the question

./sentence-classification/*sentence.pkl* # retrieved sentences for each question. pd.Dataframe with columns=['qid','sentence_sha']
```

4. Rank the most relevant sentence based on question (Kejing, Tongtong, Yan)
```
./question-answering/*code.ipynb* #code for question answering

./question-answering/*sentence.pkl* #retrived sentence for each question. pd.Dataframe with columns=['qid', 'sentence_sha', 'rank']
```

5. Human expert validate each sentence (Yimeng, Kendall). 1 if valid; 0 if maybe; -1 if irrelevant)
```
./human-correction/*code.ipynb* #qgrid visualization code

./human-correction/*sentence.pkl* #human-valided sentence for each question. pd.Dataframe with columns=['qid', 'sentence_sha', 'valid']
```


10 questions are listed in [here](https://docs.google.com/document/d/10B_VkqxDyjxjJWvS5C-q4V7p3c1F-HuLOxiu_vlWtb8/edit#)

Shared file folder is [here](https://drive.google.com/open?id=15IX5FUcb0if25J_0fZMJ-N3Ir3UkNQpK)
