# Jarak Data

Data dapat diketahui equivalensinya menggunakan penggurukan jarak dari dataset.

Sebelum itu, mari kita tarik sampel data dan beri label.


```python
import pandas as pd
import itertools
import scipy.spatial.distance as spad
```


```python
columns = ['Specimen Number', 'Eccentricity', 'Aspect Ratio', 'Elongation', 'Solidity']
df = pd.read_csv('leaf.csv', nrows=4, usecols=columns)
data = [[["","A","B","C","D"][int(x[0])]]+[round(i,2) for i in x[1:]] for x in df.values.tolist()]
df = pd.DataFrame(data, columns=df.columns)
#data = [x[1:] for x in df.values.tolist()] # Discard for REAL calculation
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
      <th>Specimen Number</th>
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
      <td>0.73</td>
      <td>1.47</td>
      <td>0.32</td>
      <td>0.99</td>
    </tr>
    <tr>
      <td>1</td>
      <td>B</td>
      <td>0.74</td>
      <td>1.53</td>
      <td>0.36</td>
      <td>0.98</td>
    </tr>
    <tr>
      <td>2</td>
      <td>C</td>
      <td>0.77</td>
      <td>1.57</td>
      <td>0.39</td>
      <td>0.98</td>
    </tr>
    <tr>
      <td>3</td>
      <td>D</td>
      <td>0.74</td>
      <td>1.46</td>
      <td>0.35</td>
      <td>0.98</td>
    </tr>
  </tbody>
</table>
</div>



## Minkowski Distance

Jarak Minkowski adalah jarak spatial dengan M sebagai parameter real dan N sebagai jumlah dimensi pada entity.

Special Case: 
+ Jika M = 1 maka bisa disebut sbg Manhattan (Cityblock) distance
+ Jika M = 2 maka bisa disebut Euclidean distance.


```python
columns = ['v1-v2', 'Manhattan distance (M=1)', 'Euclidean distance (M=2)']
data2 = [(
    a[0]+"-"+b[0],
    spad.cityblock(a[1:],b[1:]),
    spad.euclidean(a[1:],b[1:]))
    for a, b in itertools.combinations(data, 2)]
pd.DataFrame(data2,columns=columns)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>A-B</td>
      <td>0.12</td>
      <td>0.073485</td>
    </tr>
    <tr>
      <td>1</td>
      <td>A-C</td>
      <td>0.22</td>
      <td>0.128841</td>
    </tr>
    <tr>
      <td>2</td>
      <td>A-D</td>
      <td>0.06</td>
      <td>0.034641</td>
    </tr>
    <tr>
      <td>3</td>
      <td>B-C</td>
      <td>0.10</td>
      <td>0.058310</td>
    </tr>
    <tr>
      <td>4</td>
      <td>B-D</td>
      <td>0.08</td>
      <td>0.070711</td>
    </tr>
    <tr>
      <td>5</td>
      <td>C-D</td>
      <td>0.18</td>
      <td>0.120830</td>
    </tr>
  </tbody>
</table>
</div>



## Average Distance

## Weighted Distance

## Chord distance

## Mahalanobis distance

## Cosine Measure

## Pearson correlation

## Summary
