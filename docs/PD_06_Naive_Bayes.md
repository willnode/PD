# Naive Bayes Classifier


Naive Bayes Classifier adalah classifier dimana untuk setiap fitur $X$ sejumlah $n$ : 

$$ P(\operatorname{C_k}) = \frac{\left(\prod_{i=1}^n P(X_i|C_k)\right)P(C_k)}{P(X)} $$

Layman terms:

$$ \operatorname{posterior} = \frac{\operatorname{prior}\times\operatorname{likehood}}{\operatorname{evidence}} $$ 

$P$ adalah probabilitas yang muncul. Untuk data numerik $P$ adalah:

$$ P(x=v|C_k) = \frac{1}{\sqrt{2\pi\sigma^2_k}}\exp\left(-\frac{(v-\mu_k)^2}{2\sigma^2_k}\right) $$

dimana $v$ adalah nilai dalam fitur, $\sigma_k$ adalah Standar deviasi dan $\mu_k$ adalah Rata-rata untuk K (kolom)

## Langkah-Langkah Training 

#### 1. Ambil data set


```python
from sklearn import datasets
from pandas import *
from numpy import *
from math import *

from IPython.display import HTML, display; from tabulate import tabulate
def table(df): display(HTML(tabulate(df, tablefmt='html', headers='keys', showindex=False)))
```


```python
# IRIS TRAINING TABLE
iris = datasets.load_iris()
data = [list(s)+[iris.target_names[iris.target[i]]] for i,s in enumerate(iris.data)]
dataset = DataFrame(data, columns=iris.feature_names+['class']).sample(frac=0.2)
table(dataset)
```


<table>
<thead>
<tr><th style="text-align: right;">  sepal length (cm)</th><th style="text-align: right;">  sepal width (cm)</th><th style="text-align: right;">  petal length (cm)</th><th style="text-align: right;">  petal width (cm)</th><th>class     </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                5.2</td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               2.3</td><td style="text-align: right;">                3.3</td><td style="text-align: right;">               1  </td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                6.9</td><td style="text-align: right;">               2.3</td><td>virginica </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.7</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.4</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                5.2</td><td style="text-align: right;">               4.1</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.1</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                4.3</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.1</td><td style="text-align: right;">               0.1</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               2.4</td><td style="text-align: right;">                3.3</td><td style="text-align: right;">               1  </td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.4</td><td style="text-align: right;">                3.7</td><td style="text-align: right;">               1  </td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                4.7</td><td style="text-align: right;">               1.5</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.6</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               1.8</td><td>virginica </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.3</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                6.9</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.3</td><td>virginica </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                7.2</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                6  </td><td style="text-align: right;">               1.8</td><td>virginica </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                4.2</td><td style="text-align: right;">               1.3</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                4.6</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                4  </td><td style="text-align: right;">               1.2</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.3</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.1</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                7.6</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                6.6</td><td style="text-align: right;">               2.1</td><td>virginica </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               4.4</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.4</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                3.6</td><td style="text-align: right;">               1.3</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                4.4</td><td style="text-align: right;">               1.4</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.3</td><td>virginica </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.2</td><td style="text-align: right;">               1.3</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5  </td><td style="text-align: right;">               1.7</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                4.7</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.2</td><td>setosa    </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.7</td><td style="text-align: right;">               0.2</td><td>setosa    </td></tr>
</tbody>
</table>


#### 2. Sampel data untuk di tes


```python
test = [3,5,2,4]
print("sampel data: ", test)
```

    sampel data:  [3, 5, 2, 4]
    

#### 3. Identifikasi Per Grup Class Target untuk data Training


```python
dataset_classes = {}
# table per classes
for key,group in dataset.groupby('class'):
    mu_s = [group[c].mean() for c in group.columns[:-1]]
    sigma_s = [group[c].std() for c in group.columns[:-1]]
    dataset_classes[key] = [group, mu_s, sigma_s]
    print(key, "===>")
    print('Mu_s =>', array(mu_s))
    print('Sigma_s =>', array(sigma_s))
    table(group)

```

    setosa ===>
    Mu_s => [5.18333333 3.56666667 1.51666667 0.31666667]
    Sigma_s => [0.3250641  0.22509257 0.14719601 0.1602082 ]
    


<table>
<thead>
<tr><th style="text-align: right;">  sepal length (cm)</th><th style="text-align: right;">  sepal width (cm)</th><th style="text-align: right;">  petal length (cm)</th><th style="text-align: right;">  petal width (cm)</th><th>class  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.6</td><td>setosa </td></tr>
<tr><td style="text-align: right;">                5.2</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3.9</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.4</td><td>setosa </td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.2</td><td>setosa </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                1.7</td><td style="text-align: right;">               0.3</td><td>setosa </td></tr>
</tbody>
</table>


    versicolor ===>
    Mu_s => [5.89166667 2.76666667 4.13333333 1.26666667]
    Sigma_s => [0.52476546 0.33393884 0.46188022 0.21461735]
    


<table>
<thead>
<tr><th style="text-align: right;">  sepal length (cm)</th><th style="text-align: right;">  sepal width (cm)</th><th style="text-align: right;">  petal length (cm)</th><th style="text-align: right;">  petal width (cm)</th><th>class     </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.1</td><td style="text-align: right;">               1.3</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                4.4</td><td style="text-align: right;">               1.2</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.4</td><td style="text-align: right;">                3.7</td><td style="text-align: right;">               1  </td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.6</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                3.5</td><td style="text-align: right;">               1  </td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                3.9</td><td style="text-align: right;">               1.1</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6.8</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.8</td><td style="text-align: right;">               1.4</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                4.4</td><td style="text-align: right;">               1.4</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                4  </td><td style="text-align: right;">               1.2</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               2.3</td><td style="text-align: right;">                3.3</td><td style="text-align: right;">               1  </td><td>versicolor</td></tr>
</tbody>
</table>


    virginica ===>
    Mu_s => [6.61666667 3.13333333 5.58333333 2.06666667]
    Sigma_s => [0.7790826  0.34465617 0.59670814 0.23868326]
    


<table>
<thead>
<tr><th style="text-align: right;">  sepal length (cm)</th><th style="text-align: right;">  sepal width (cm)</th><th style="text-align: right;">  petal length (cm)</th><th style="text-align: right;">  petal width (cm)</th><th>class    </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               3.3</td><td style="text-align: right;">                6  </td><td style="text-align: right;">               2.5</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                5.5</td><td style="text-align: right;">               1.8</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               1.9</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.4</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                6.5</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.2</td><td style="text-align: right;">               2  </td><td>virginica</td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               2  </td><td>virginica</td></tr>
<tr><td style="text-align: right;">                5.9</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               1.8</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.3</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                6.8</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.1</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                6.7</td><td style="text-align: right;">               2.2</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               1.8</td><td>virginica</td></tr>
<tr><td style="text-align: right;">                7.9</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                6.4</td><td style="text-align: right;">               2  </td><td>virginica</td></tr>
</tbody>
</table>


#### 5. Hitung Probabilitas Prior dan Likehood

WIP: Probabilitas Evidence masukkan ke hitungan


```python

def numericalPriorProbability(v, mu, sigma):
    return (1.0/sqrt(2 * pi * (sigma ** 2))*exp(-((v-mu)**2)/(2*(sigma**2))))

def categoricalProbability(sample,universe):
    return sample.shape[0]/universe.shape[0]

Ps = ([[y]+[numericalPriorProbability(x, d[1][i], d[2][i]) for i,x in enumerate(test)]+
          [categoricalProbability(d[0],dataset)] for y,d in dataset_classes.items()])

table(DataFrame(Ps, columns=["classes"]+["P( %d | C )" % d for d in test]+["P( C )"]))
```


<table>
<thead>
<tr><th>classes   </th><th style="text-align: right;">  P( 3 | C )</th><th style="text-align: right;">  P( 5 | C )</th><th style="text-align: right;">  P( 2 | C )</th><th style="text-align: right;">  P( 4 | C )</th><th style="text-align: right;">  P( C )</th></tr>
</thead>
<tbody>
<tr><td>setosa    </td><td style="text-align: right;"> 1.96232e-10</td><td style="text-align: right;"> 2.77721e-09</td><td style="text-align: right;"> 0.0123515  </td><td style="text-align: right;">4.13093e-115</td><td style="text-align: right;">     0.2</td></tr>
<tr><td>versicolor</td><td style="text-align: right;"> 1.93812e-07</td><td style="text-align: right;"> 2.31644e-10</td><td style="text-align: right;"> 2.01329e-05</td><td style="text-align: right;">1.11579e-35 </td><td style="text-align: right;">     0.4</td></tr>
<tr><td>virginica </td><td style="text-align: right;"> 1.07096e-05</td><td style="text-align: right;"> 4.94165e-07</td><td style="text-align: right;"> 9.87125e-09</td><td style="text-align: right;">9.46396e-15 </td><td style="text-align: right;">     0.4</td></tr>
</tbody>
</table>


#### 6. Rank & Tarik Kesimpulan


```python
Pss = ([[r[0], prod(r[1:])] for r in Ps])
PDss = DataFrame(Pss, columns=['class', 'probability']).sort_values('probability')[::-1]
table(PDss)
```


<table>
<thead>
<tr><th>class     </th><th style="text-align: right;">  probability</th></tr>
</thead>
<tbody>
<tr><td>virginica </td><td style="text-align: right;"> 1.97766e-34 </td></tr>
<tr><td>versicolor</td><td style="text-align: right;"> 4.03412e-57 </td></tr>
<tr><td>setosa    </td><td style="text-align: right;"> 5.56132e-136</td></tr>
</tbody>
</table>



```python
print("Prediksi Bayes untuk", test, "adalah", PDss.values[0,0])
```

    Prediksi Bayes untuk [3, 5, 2, 4] adalah virginica
    

## Real Test

diberikan variabel `dataset` dan `dataset_classes` yang sebagai 'data training' kita, mari kita lakukan itu untuk real Iris data:


```python
# ONE FUNCTION FOR CLASSIFIER

def predict(sampel):
    priorLikehoods = ([[y]+[numericalPriorProbability(x, d[1][i], d[2][i]) for i,x in enumerate(sampel)]+
          [categoricalProbability(d[0],dataset)] for y,d in dataset_classes.items()])
    products = ([[r[0], prod(r[1:])] for r in priorLikehoods])
    result = DataFrame(products, columns=['class', 'probability']).sort_values('probability')[::-1]
    return result.values[0,0]

dataset_test = DataFrame([list(d)+[predict(d[:4])] for d in data], columns=list(dataset.columns)+['predicted class (by predict())'])
table(dataset_test)
```


<table>
<thead>
<tr><th style="text-align: right;">  sepal length (cm)</th><th style="text-align: right;">  sepal width (cm)</th><th style="text-align: right;">  petal length (cm)</th><th style="text-align: right;">  petal width (cm)</th><th>class     </th><th>predicted class (by predict())  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.7</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.6</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.6</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3.9</td><td style="text-align: right;">                1.7</td><td style="text-align: right;">               0.4</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.6</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.3</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.4</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.1</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3.7</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.1</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.3</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.1</td><td style="text-align: right;">               0.1</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               4  </td><td style="text-align: right;">                1.2</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               4.4</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.4</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3.9</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.4</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.3</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                1.7</td><td style="text-align: right;">               0.3</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.3</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.7</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.7</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.4</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.6</td><td style="text-align: right;">               3.6</td><td style="text-align: right;">                1  </td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.3</td><td style="text-align: right;">                1.7</td><td style="text-align: right;">               0.5</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.9</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.4</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.2</td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.2</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.7</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.4</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.2</td><td style="text-align: right;">               4.1</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.1</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               4.2</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                1.2</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               3.6</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.1</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.4</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.3</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.5</td><td style="text-align: right;">               2.3</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.3</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.4</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                1.3</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.5</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.6</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                1.9</td><td style="text-align: right;">               0.4</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.8</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.3</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                1.6</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                4.6</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5.3</td><td style="text-align: right;">               3.7</td><td style="text-align: right;">                1.5</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               3.3</td><td style="text-align: right;">                1.4</td><td style="text-align: right;">               0.2</td><td>setosa    </td><td>setosa                          </td></tr>
<tr><td style="text-align: right;">                7  </td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                4.7</td><td style="text-align: right;">               1.4</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.9</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.3</td><td style="text-align: right;">                4  </td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.5</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.6</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               3.3</td><td style="text-align: right;">                4.7</td><td style="text-align: right;">               1.6</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               2.4</td><td style="text-align: right;">                3.3</td><td style="text-align: right;">               1  </td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.6</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.6</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.2</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                3.9</td><td style="text-align: right;">               1.4</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               2  </td><td style="text-align: right;">                3.5</td><td style="text-align: right;">               1  </td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.9</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.2</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               2.2</td><td style="text-align: right;">                4  </td><td style="text-align: right;">               1  </td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.7</td><td style="text-align: right;">               1.4</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                3.6</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                4.4</td><td style="text-align: right;">               1.4</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                4.1</td><td style="text-align: right;">               1  </td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.2</td><td style="text-align: right;">               2.2</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                3.9</td><td style="text-align: right;">               1.1</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.9</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                4.8</td><td style="text-align: right;">               1.8</td><td>versicolor</td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4  </td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.7</td><td style="text-align: right;">               1.2</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.3</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.6</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.4</td><td style="text-align: right;">               1.4</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.8</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.8</td><td style="text-align: right;">               1.4</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5  </td><td style="text-align: right;">               1.7</td><td>versicolor</td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                3.5</td><td style="text-align: right;">               1  </td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.4</td><td style="text-align: right;">                3.8</td><td style="text-align: right;">               1.1</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.4</td><td style="text-align: right;">                3.7</td><td style="text-align: right;">               1  </td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                3.9</td><td style="text-align: right;">               1.2</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               1.6</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.4</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.6</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                4.7</td><td style="text-align: right;">               1.5</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               2.3</td><td style="text-align: right;">                4.4</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.1</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                4  </td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                4.4</td><td style="text-align: right;">               1.2</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.6</td><td style="text-align: right;">               1.4</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                4  </td><td style="text-align: right;">               1.2</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5  </td><td style="text-align: right;">               2.3</td><td style="text-align: right;">                3.3</td><td style="text-align: right;">               1  </td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                4.2</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.2</td><td style="text-align: right;">               1.2</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.2</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.2</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                4.3</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.1</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                3  </td><td style="text-align: right;">               1.1</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.1</td><td style="text-align: right;">               1.3</td><td>versicolor</td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               3.3</td><td style="text-align: right;">                6  </td><td style="text-align: right;">               2.5</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               1.9</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.1</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.9</td><td style="text-align: right;">               2.1</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                5.6</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.5</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.2</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.6</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                6.6</td><td style="text-align: right;">               2.1</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                4.9</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                4.5</td><td style="text-align: right;">               1.7</td><td>virginica </td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                7.3</td><td style="text-align: right;">               2.9</td><td style="text-align: right;">                6.3</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                5.8</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.2</td><td style="text-align: right;">               3.6</td><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.5</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.5</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               2  </td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                5.3</td><td style="text-align: right;">               1.9</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.8</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.5</td><td style="text-align: right;">               2.1</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                5  </td><td style="text-align: right;">               2  </td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               2.4</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                5.3</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.5</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.5</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                6.7</td><td style="text-align: right;">               2.2</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                6.9</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               2.2</td><td style="text-align: right;">                5  </td><td style="text-align: right;">               1.5</td><td>virginica </td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.9</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               2  </td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                6.7</td><td style="text-align: right;">               2  </td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.3</td><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.1</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.2</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                6  </td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.2</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                4.8</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.9</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.1</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.2</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.8</td><td style="text-align: right;">               1.6</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.4</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                6.1</td><td style="text-align: right;">               1.9</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                7.9</td><td style="text-align: right;">               3.8</td><td style="text-align: right;">                6.4</td><td style="text-align: right;">               2  </td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.2</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               2.8</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               1.5</td><td>virginica </td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.6</td><td style="text-align: right;">                5.6</td><td style="text-align: right;">               1.4</td><td>virginica </td><td>versicolor                      </td></tr>
<tr><td style="text-align: right;">                7.7</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                6.1</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.4</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.4</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                5.5</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6  </td><td style="text-align: right;">               3  </td><td style="text-align: right;">                4.8</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.9</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                5.4</td><td style="text-align: right;">               2.1</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                5.6</td><td style="text-align: right;">               2.4</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.9</td><td style="text-align: right;">               3.1</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                5.8</td><td style="text-align: right;">               2.7</td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               1.9</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.8</td><td style="text-align: right;">               3.2</td><td style="text-align: right;">                5.9</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3.3</td><td style="text-align: right;">                5.7</td><td style="text-align: right;">               2.5</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.7</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.2</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.3</td><td style="text-align: right;">               2.5</td><td style="text-align: right;">                5  </td><td style="text-align: right;">               1.9</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.5</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.2</td><td style="text-align: right;">               2  </td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                6.2</td><td style="text-align: right;">               3.4</td><td style="text-align: right;">                5.4</td><td style="text-align: right;">               2.3</td><td>virginica </td><td>virginica                       </td></tr>
<tr><td style="text-align: right;">                5.9</td><td style="text-align: right;">               3  </td><td style="text-align: right;">                5.1</td><td style="text-align: right;">               1.8</td><td>virginica </td><td>virginica                       </td></tr>
</tbody>
</table>


#### Kesimpulan


```python

corrects = dataset_test.loc[dataset_test['class'] == dataset_test['predicted class (by predict())']].shape[0]
print('Prediksi Training Bayes: %d of %d == %f %%' % (corrects, len(data), corrects / len(data) * 100))
```

    Prediksi Training Bayes: 144 of 150 == 96.000000 %
    
