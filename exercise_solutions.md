# Data Preparation

### Bacteria DataFrame indexing:

```python
bacteria_data[bacteria_data.phylum.str.endswith('bacteria') & (bacteria_data.value>1000)]
```

### Chunking CSV files:

```python
mean_tissue = {chunk.Taxon[0]:chunk.Tissue.mean() for chunk in data_chunks}
```

### Filling missing values in microbiome data:

```python
missing_sample.fillna(missing_sample.groupby('Taxon').transform('mean'))
```

# Density Estimation

### Chopsticks effectiveness estimation

```python
chopsticks.hist('Food_Pinching_Efficiency', by='Chopstick_Length', sharex=True, bins=6)
```

- the easy way:

```python
chopsticks.groupby('Chopstick_Length').apply(lambda x: norm.fit(x.Food_Pinching_Efficiency))
```

- the hard way:

```python
neg_norm_like = lambda theta, x: 0.5*len(x)*np.log(theta[1]**2) + 0.5*((x-theta[0])**2).sum()/(theta[1]**2)

fpe_by_len = chopsticks.groupby('Chopstick_Length')['Food_Pinching_Efficiency']
fpe_by_len.apply(lambda x: fmin(neg_norm_like, np.array([20,5]), args=(x,)))
```

# Regression models

### Multivariate regression for Olympic medals

```python
b1, b2, b0 = fmin(poisson_loglike, [0,1,0], args=(medals[['log_population', 'oecd']].assign(intercept=1).values, 
                                            medals.medals.values))
b0, b1, b2
```

### Olympic medals model selection

```python
models = ('medals.log_population', 
          'medals.log_population + medals.oecd', 
          'medals.log_population * medals.oecd')

aic_values = {}

for model in models:
    
    X = dmatrix(model)
    k = X.shape[1]
    params = fmin(poisson_loglike, np.ones(k), args=(X, medals.medals.values))
    loglike = poisson_loglike(params, X, medals.medals.values)
    aic = -2 * loglike + 2*k
    aic_values[model] = aic
    
aic_values
```

# Resampling & Missing Data

### K-fold salmon

Non-generator k-fold

```python
k = 5
n = salmon.shape[0]/k

test_subsets = []
remaining_dataset = salmon.copy()
for i in range(k):
    test = remaining_dataset.sample(n=n)
    test_subsets.append(test)
    remaining_dataset = remaining_dataset.drop(test.index)
    
train_subsets = [salmon.drop(t.index) for t in test_subsets]
```

Generator for k-fold

```python
def kfold(data, k=5):
    
    n = data.shape[0]/k
    remaining_dataset = data.copy()
    for i in range(k):
        test = remaining_dataset.sample(n=n)
        remaining_dataset = remaining_dataset.drop(test.index)
        
        yield test, data.drop(test.index)
```

scikit-learn KFold

```python
from sklearn.cross_validation import cross_val_score, KFold

nfolds = 5
fig, axes = plt.subplots(1, nfolds, figsize=(14,4))
for i, fold in enumerate(KFold(len(salmon), n_folds=nfolds, 
                               shuffle=True)):
    training, validation = fold
    y, x = salmon.values[training].T
    axes[i].plot(x, y, 'ro')
    y, x = salmon.values[validation].T
    axes[i].plot(x, y, 'bo')
    
plt.tight_layout()
```

### Poisson regression confidence intervals

```python
from scipy.optimize import fmin

medals = pd.read_csv('../data/medals.csv')

poisson_loglike = lambda beta, X, y: -(-np.exp(X.dot(beta)) + y*X.dot(beta)).sum()

R = 1000

def bootstrap_estimates(R):
    
    for i in range(R):
        bootstrap_medals = medals.sample(n=medals.shape[0], replace=True)
        X = bootstrap_medals[['log_population']].assign(intercept=1).values
        y = bootstrap_medals.medals.values
        yield fmin(poisson_loglike, [0,1], args=(X, y), disp=False)
        
params = list(bootstrap_estimates(R))

np.sort(np.transpose(params))[:, [25, 975]].round(2)
```

Plots of bootstrapped fits

```python
ax = medals.plot(x='log_population', y='medals', kind='scatter')

xvals = np.arange(12, 22)
for b1,b0 in params:
    ax.plot(xvals, np.exp(b0 + b1*xvals), 'b-', alpha=0.02)
```

### Titanic imputation

```python
titanic = pd.read_excel("../data/titanic.xls", "titanic")

from sklearn import linear_model

# Drop the single missing fare
impute_dataset = titanic.dropna(subset=['fare'])
# Mask for missing age
missing_age = impute_dataset.age.isnull()

# Models for imputing age
models = (['pclass', 'fare', 'parch', 'sibsp'],
         ['pclass', 'parch', 'sibsp'],
         ['fare', 'parch', 'sibsp'],
         ['pclass', 'sibsp'],
         ['fare', 'sibsp'])

age_imp = []

for model in models:
    
    regmod = linear_model.LinearRegression()
    regmod.fit(impute_dataset.loc[~missing_age, model], impute_dataset.loc[~missing_age, 'age'])
    imputed = regmod.predict(impute_dataset.loc[missing_age, model])
    age_imp.append(imputed)
```

Now fit the models

```python
coefficients = []

impute_dataset = impute_dataset.assign(male=impute_dataset.sex=='male')

for imputes in age_imp:
    
    regr = linear_model.LogisticRegression()
    
    X = impute_dataset[['male', 'age', 'fare']].copy()
    X.loc[missing_age, 'age'] = imputes
    y = impute_dataset['survived']
    regr.fit(X, y)
    coefficients.append(regr.coef_.ravel())
    
pd.DataFrame(coefficients).mean()
```
