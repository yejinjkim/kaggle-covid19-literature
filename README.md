# kaggle-covid19-literature

Requirements
```
pip install tqdm boto3 requests regex sentencepiece sacremoses
pip install transformers
pip install -U git+https://github.com/dgunning/cord19.git
pip install langdetect
pip install pandas
pip install tqdm 
pip install snorkel
```

![Overview](https://github.com/yejinjkim/kaggle-covid19-literature/blob/master/overview.png)
1. Raw data
```
./raw-data # the literature files provided from Kaggle
```

2. Pseudo labelling with keywords and Snorkel (Kanglin, Shana, Qian)
```
./snorkel-pseudo-label/Snorkel_pseudo_label.ipynb* #code for the pseudo label using Snorkel. 
```
3. Retrieve relevant sentences (Kanglin, Yejin)
```
./sentence-classification/bert_classification.ipynb #code for sentence classification to retreive relevant sentences to the question
./sentence-classification/evaluation_BERT.ipynb #code for re-ranking sentences based on keywords and reference sentence similarity
```

4. Question-answering (Kejing, Tongtong, Yan)
```
./question-answering/*code.ipynb* #code for question answering

./question-answering/*sentence.pkl* #retrived sentence for each question. pd.Dataframe with columns=['qid', 'sentence_sha', 'rank']
```

5. Visualization (Yimeng, Kendall)
```
./human-correction/*code.ipynb* #qgrid visualization code

```



10 questions are listed in [here](https://docs.google.com/document/d/10B_VkqxDyjxjJWvS5C-q4V7p3c1F-HuLOxiu_vlWtb8/edit#)

Shared file folder is [here](https://drive.google.com/open?id=15IX5FUcb0if25J_0fZMJ-N3Ir3UkNQpK)
