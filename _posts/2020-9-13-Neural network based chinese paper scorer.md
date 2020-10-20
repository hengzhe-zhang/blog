---
layout: post
title: Neural network based chinese paper scorer (Transformer+XGBoost)
---
**This project is only used for education purposes, we should not use it in real-world.**

As a teaching assistant, I always have to review a lot of papers, which makes me feel under great pressure. Thus, in order to relieve myself from repetitive work, I build this project and attempt to use machine learning to automate the scoring process.

In contrast to the traditional neural network based scorer, this scorer is based on a traditional machine learning technique, XGBoost. The reason behind this is that XGBoost is easier to train compared with neural network in terms of resources and time consumption.

The basic idea is rather simple, we can extract text from PDF documents, and then use Transformer to extract the document embedding from the documents. After that, we can use XGBoost to train based on document embedding and document scoring.

### Document extraction
Due to the privacy policy, in this project, I will not use any private papers. Thus, I downloaded several papers from CNKI as training data. I marked the paper downloaded from top-notch Chinese journal as 100 points and other papers as 60 points.

Because all documents are PDF format, we should convert those files to text first. This is rather simple by using the package "textract". It should be noted that the current implementation of textract may raise errors when converting some documents. Thus, I slightly modified the source code to avoid that issue. The modified version of "textract" can be seen in the Github repository of this project.


```python
from tqdm import tqdm

import textract

def file_to_text(pdf):
    text = textract.process(pdf)
    return text.decode('utf8')


def data_preparation(train):
    d = []
    error = 0
    for i in ['A','B']:
        path = f'pdf/{"train" if train else "test"}/{i}'
        for f in tqdm(os.listdir(path)):
            score = 100 if i=='A' else 60
            try:
                text = file_to_text(os.path.join(path, f))
            except Exception as e:
                error += 1
                print(e)
                continue
            d.append({
                'text': text,
                'score': score
            })
    frame = pd.DataFrame(d)
    print('Total error', error)
    return frame

train_data=data_preparation(train=True)
```

    100%|██████████| 20/20 [00:12<00:00,  1.54it/s]
    100%|██████████| 20/20 [00:09<00:00,  2.13it/s]
    

    Total error 0
    

### Model training
After extracting the documents, we get a table that contains training text and training score. Then, we can use the sentence transformer to encode our text. The theory of transformer is not the focus point of this paper, thus we will omit it. The only thing we need to remember is that the Transformer can encode a paragraph to a fixed length vector by using the pre-trained neural network.

In the final step, we only need to fit the embedded text and the corresponding training score by using XGBoost. We use PCA to reduce the dimension of the embedding vector because our training data is too small.


```python

import pandas as pd
import os
from harvesttext import HarvestText
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def batch_embedding(corpus):
    ht0 = HarvestText()
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    corpus_embeddings = model.encode(list(map(lambda x: ht0.clean_text(x), corpus.tolist())))
    return corpus_embeddings

def train_task(data):
    print('data', data.shape)
    embedding = batch_embedding(data['text'])
    pipe = Pipeline([('PCA', PCA(n_components=24)), ('xgb', XGBRegressor())])
    score = cross_validate(pipe, embedding, data['score'], scoring='neg_mean_absolute_error', cv=3)
    print(score)
    pipe.fit(embedding, data['score'])
    return pipe
```


```python
mode=train_task(train_data)
```

    data (40, 2)
    {'fit_time': array([0.10973072, 0.10060215, 0.1289463 ]), 'score_time': array([0.00138736, 0.00129271, 0.00121737]), 'test_score': array([-3.50952148e-04, -1.92788931e-04, -2.77989003e+01])}
    

### Model validation
In order to validate our model, next, we will apply our model on the test data.


```python
test_data=data_preparation(train=False)
```

    100%|██████████| 20/20 [00:07<00:00,  2.70it/s]
    100%|██████████| 20/20 [00:07<00:00,  2.84it/s]
    

    Total error 0
    

Finally, we find that the model can generalize well on the testing data. It should be noted that in this project, we set the same score for those papers that were published in the same journal. For a more accurate model, we also can manually mark the score for each paper, and then we can get a more accurate score for each new paper.


```python
from sklearn.metrics import mean_absolute_error
embedding = batch_embedding(test_data['text'])
print(mean_absolute_error(test_data['score'],mode.predict(embedding)))
```

    3.0001313209533693
    
