# Jarak Data

Data dapat diketahui equivalensinya menggunakan penggurukan jarak dari dataset.

Sebelum itu, mari kita tarik sampel data dan beri label.


```python
from pandas import *
import itertools
import scipy.spatial.distance as spad
```


```python
columns = ['Specimen Number', 'Eccentricity', 'Aspect Ratio', 'Elongation', 'Solidity']
df = read_csv('leaf.csv', nrows=4, usecols=columns)
data = [[["","A","B","C","D"][int(x[0])]]+[round(i*2,2) for i in x[1:]] for x in df.values.tolist()]
df = DataFrame(data, columns=['Class']+columns[1:])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Eccentricity</th>
      <th>Aspect Ratio</th>
      <th>Elongation</th>
      <th>Solidity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>A</td>
      <td>1.45</td>
      <td>2.95</td>
      <td>0.65</td>
      <td>1.97</td>
    </tr>
    <tr>
      <td>1</td>
      <td>B</td>
      <td>1.48</td>
      <td>3.05</td>
      <td>0.72</td>
      <td>1.96</td>
    </tr>
    <tr>
      <td>2</td>
      <td>C</td>
      <td>1.53</td>
      <td>3.15</td>
      <td>0.78</td>
      <td>1.96</td>
    </tr>
    <tr>
      <td>3</td>
      <td>D</td>
      <td>1.48</td>
      <td>2.92</td>
      <td>0.71</td>
      <td>1.95</td>
    </tr>
  </tbody>
</table>
</div>



## Minkowski Distance

Jarak Minkowski adalah jarak spatial antara dua record ($x$ dan $y$) dengan $m$ sebagai parameter real dan $n$ sebagai jumlah dimensi pada entity.

$$ d_{\operatorname{minkowski}} = \left(\sum_{i=1}^{n}|x_{i}-y_{i}|^{m}\right)^{\frac{1}{m}},m\geq 1$$

Special Case: 
+ Jika M = 1 maka bisa disebut Manhattan (Cityblock) distance

$$ d_{\operatorname{manhattan}} = \sum_{i=1}^{n}|x_{i}-y_{i}|$$

+ Jika M = 2 maka bisa disebut Euclidean distance.

$$ d_{\operatorname{euclidean}} = \sqrt{\sum_{i=1}^{n}|x_{i}-y_{i}|^{2}} $$

Ilustrasi menggambarkan hitungan Cityblock (garis putus-putus) dan Euclidean (garis lurus) secara visual:

![](https://www.researchgate.net/profile/Jose_Palma6/publication/229342959/figure/fig5/AS:300865354256401@1448743298665/City-block-distances-dashed-lines-between-Y-a-and-Y-b-in-a-2-dimensional-space-Note.png)


```python
columns = ['v1-v2', 'Manhattan distance (M=1)', 'Euclidean distance (M=2)', 'Minkowski distance at M=3']
data2 = [(
    "{} - {}".format(a[0],b[0]),
    spad.cityblock(a[1:],b[1:]),
    spad.euclidean(a[1:],b[1:]),
    spad.minkowski(a[1:],b[1:],3),
    )
    for a, b in itertools.combinations(data, 2)]
DataFrame(data2,columns=columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1-v2</th>
      <th>Manhattan distance (M=1)</th>
      <th>Euclidean distance (M=2)</th>
      <th>Minkowski distance at M=3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>A - B</td>
      <td>0.21</td>
      <td>0.126095</td>
      <td>0.111091</td>
    </tr>
    <tr>
      <td>1</td>
      <td>A - C</td>
      <td>0.42</td>
      <td>0.251794</td>
      <td>0.220426</td>
    </tr>
    <tr>
      <td>2</td>
      <td>A - D</td>
      <td>0.14</td>
      <td>0.076158</td>
      <td>0.065265</td>
    </tr>
    <tr>
      <td>3</td>
      <td>B - C</td>
      <td>0.21</td>
      <td>0.126886</td>
      <td>0.110275</td>
    </tr>
    <tr>
      <td>4</td>
      <td>B - D</td>
      <td>0.15</td>
      <td>0.130767</td>
      <td>0.130039</td>
    </tr>
    <tr>
      <td>5</td>
      <td>C - D</td>
      <td>0.36</td>
      <td>0.245764</td>
      <td>0.232918</td>
    </tr>
  </tbody>
</table>
</div>



## Average Distance


```python

```


```python
columns = ['v1-v2', 'Average Distance', 'Euclidean distance (M=2)', 'Minkowski M=3']
data2 = [(
    "{} - {}".format(a[0],b[0]),
    spad.cityblock(a[1:],b[1:]),
    spad.euclidean(a[1:],b[1:]),
    spad.euclidean(a[1:],b[1:]),
    )
    for a, b in itertools.combinations(data, 2)]
DataFrame(data2,columns=columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1-v2</th>
      <th>Average Distance</th>
      <th>Euclidean distance (M=2)</th>
      <th>Minkowski M=3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>A - B</td>
      <td>0.21</td>
      <td>0.126095</td>
      <td>0.126095</td>
    </tr>
    <tr>
      <td>1</td>
      <td>A - C</td>
      <td>0.42</td>
      <td>0.251794</td>
      <td>0.251794</td>
    </tr>
    <tr>
      <td>2</td>
      <td>A - D</td>
      <td>0.14</td>
      <td>0.076158</td>
      <td>0.076158</td>
    </tr>
    <tr>
      <td>3</td>
      <td>B - C</td>
      <td>0.21</td>
      <td>0.126886</td>
      <td>0.126886</td>
    </tr>
    <tr>
      <td>4</td>
      <td>B - D</td>
      <td>0.15</td>
      <td>0.130767</td>
      <td>0.130767</td>
    </tr>
    <tr>
      <td>5</td>
      <td>C - D</td>
      <td>0.36</td>
      <td>0.245764</td>
      <td>0.245764</td>
    </tr>
  </tbody>
</table>
</div>



## Weighted Distance

## Chord distance

## Mahalanobis distance

## Cosine Measure

## Pearson correlation

## Summary
