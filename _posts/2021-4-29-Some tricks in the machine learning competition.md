---
layout: post
title: Some tricks in the machine learning competition 
---

There are many tricks that can help us get a good score in a machine learning competition. As a result, in this article, I will focus on presenting some small tricks that are commonly used in machine learning challenges but are rarely mentioned in textbooks. As a result, I hope that by writing this article, newcomers to the field of machine learning will be able to benefit from such intriguing tricks.

### Data Preprocessing
   
1. Missing Data Analysis 

	Analyzing missing data in the data is a very useful way to identify useless features and design imputation policies in the subsequent steps. To accomplish this, we can use the "missingno" package.

	```python
	import missingno as msno

	msno.bar(info)
	```

2. Remove the column with an excessive amount of missing data 
	
	After analyzing the missing data, we can use the following function to remove features with an excessive number of missing values.

	```python
	def filter_col_by_nan(df, ratio=0.05):
	    cols = []
	    for col in df.columns:
	        if df[col].isna().mean() >= (1 - ratio):
	            cols.append(col)
	    return cols
	```

### Feature Construction

1. Creating aggregated features with aggregation functions

   In some cases, the training data contains multiple tables that must be merged into a single table before being fed into the machine learning system. In most cases, we prefer to do this with aggregation features. However, if we use the pandas function, this will be a much simpler process. The aggregation function is a very useful tool for accomplishing this goal.

   As shown in the following example, after aggregating the salary feature using some aggregation functions, such as summation or averaging, we can get multiple aggregated salary features, which we can merge into our main table.

	```python
	df = pd.DataFrame({"Job": ["Programmer", "Programmer", "Writer", "Writer"], "Salary": [1, 2, 3, 4]})
	df.groupby(["Job"]).agg({"Salary": ["sum", "mean"]})
	```

	```
	Salary         sum mean               
	Programmer      3  1.5
	Writer          7  3.5
	```

2. Use pivotal tables to build aggregation features 

	Although the aggregation function can merge numerical features, dealing with categorical features is still difficult. For example, we can use the averaging function to aggregate the "salary" feature. However, we cannot directly use that function to aggregate the "job" feature. To deal with such a situation, we can use the pivotal table to aggregate those features.

	In the following example, we obtain the aggregated salary not only based on job but also based on location. We can build aggregated features based on categorical features as well as numeric features in this way.

	```python
	df = pd.DataFrame({"Job": ["Programmer", "Programmer", "Programmer", "Programmer", "Programmer",
	                           "Writer", "Writer", "Writer", "Writer"],
	                   "Location": ["Shanghai", "NewYork", "NewYork", "Shanghai",
	                                "Shanghai", "NewYork", "Shanghai", "Shanghai",
	                                "NewYork"],
	                   "Salary": [1, 2, 2, 3, 3, 4, 5, 6, 7]})

	table = pd.pivot_table(df, values='Salary', index=['Job'],
	                       columns=['Location'], aggfunc=np.sum)
	```

	```
	Location    NewYork  Shanghai                    
	Programmer        4         7
	Writer           11        11
	```

	The following are some examples of common aggregation functions.

	```python
	['max', 'min', 'mean', 'count', 'nunique', 'sum']
	```

3. Creating polynomial features 
	
	Polynomial features are a simple and effective way to improve feature quality. For example, if we have two income sources, bonus and salary, we can combine them into a single item as the total income feature.

	```python
	df = pd.DataFrame({"Job": ["Programmer", "Programmer", "Writer", "Writer"],
	                   "Salary": [1, 2, 3, 4], "Bonus": [1, 2, 3, 4]})
	df[["Salary", "Bonus"]].sum(1)
	```

​	Aside from additive features, division features are also important. Based on a division feature, for example, we can compute the average hourly wage.

	```python
	df = pd.DataFrame({"Job": ["Programmer", "Programmer", "Writer", "Writer"],
	                   "Salary": [1, 2, 3, 4], "Working Time": [1, 1, 1, 0.5]})
	df["Salary"] / df["Working Time"]
	```

​	If our dataset has a time feature, we can use subtraction to create some time interval features.

	```python
	df = pd.DataFrame({
	    "StartDate": ["2020-01-01", "2020-01-01"],
	    "EndDate": ["2020-01-05", "2020-01-10"],
	})
	days = (pd.to_datetime(df["EndDate"]) - pd.to_datetime(df["StartDate"])).dt.days
	```

4. Combining uncommon levels 

	Combining rare levels is a useful process for decision tree-based learning algorithms to improve model performance. Although it appears counterintuitive because combining different levels into a single value implies information loss. However, an intuitive way to understand this concept is that the decision tree will only split those features that will result in a high information gain after splitting. However, the information gain is not as great for those rare levels. As a result, even though these features may be useful, they will never be used. To deal with this situation, we can group those rare level features into a single group, as shown in the code below.

	```python
	def combination(df):
	    for col in ['Column']:
	        df[col + '_COUNT'] = df[col].map(df[col].value_counts())
	        col_idx = df[col].value_counts()
	        for idx in col_idx[col_idx < 10].index:
	            df[col] = df[col].replace(idx, -1)
	```

5. Encoding of labels

   Machine learning textbooks cover label encoding and one-hot encoding thoroughly. As a result, it is no longer necessary to introduce these fundamental concepts. In this article, I'd like to discuss a simple method for creating label encoded features.

	```python
	df = pd.DataFrame({"Job": ["Programmer", "Programmer", "Writer", "Writer"]})
	pd.factorize(df["Job"])[0]
	# array([0, 0, 1, 1], dtype=int64)
	```

### Model Construction

1. If we use XGBoost or LightGBM as our learning model, we should take advantage of the early stopping feature offered by these packages. However, if we want to use that feature, we must split a validation set, which may result in the waste of training data. What needs to be done to address this issue? The cross-validation scheme is an intelligent solution. We can average the prediction results with different validation sets using the function below. This way, you can avoid having to manually tune the iteration parameter of XGBoost while not wasting any training data.

	```python
	def k_fold_serachParmaters(model, train_val_data, train_val_kind, test_kind):
	    mean_f1 = 0
	    mean_f1Train = 0
	    n_splits = 5

	    cat_features = []

	    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
	    pred_Test = np.zeros(len(test_kind))
	    for train, test in sk.split(train_val_data, train_val_kind):
	        x_train = train_val_data.iloc[train]
	        y_train = train_val_kind.iloc[train]
	        x_test = train_val_data.iloc[test]
	        y_test = train_val_kind.iloc[test]

	        model.fit(x_train, y_train,
	                  eval_set=[(x_test, y_test)],
	                  categorical_feature=cat_features,
	                  early_stopping_rounds=100,
	                  verbose=False)

	        pred = model.predict(x_test)
	        fper_class = f1_score(y_test, pred)

	        pred_Train = model.predict(x_train)
	        pred_Test += model.predict_proba(test_kind)[:, 1] / n_splits
	        fper_class_train = f1_score(y_train, pred_Train)

	        mean_f1 += fper_class / n_splits
	        mean_f1Train += fper_class_train / n_splits
	    return mean_f1, pred_Test
	```

### Conclusion 
This article only introduces some general ideas for improving our model's performance in a machine learning competition. However, in order to win a competition, we must carefully analyze the training data and create some magical features based on our domain knowledge. Nonetheless, I hope this article can provide some new insights for machine learning newcomers on how to improve their scores in a machine learning competition.
