# Seleksi Sampel

Mari kita buat beberapa sampel


```python
import pandas as pd
import statistics, itertools
from IPython.display import HTML, display
from tabulate import tabulate
import scipy.spatial.distance as spad

def table(df): display(HTML(tabulate(df, tablefmt='html', headers='keys', showindex=False)))
```


```python
df = pd.read_csv('outlier.csv', usecols=['user_id', 'pause_video', 'play_video', 'seek_video', 'stop_video'], nrows=20)
table(df)
```


<table>
<thead>
<tr><th style="text-align: right;">  user_id</th><th style="text-align: right;">  pause_video</th><th style="text-align: right;">  play_video</th><th style="text-align: right;">  seek_video</th><th style="text-align: right;">  stop_video</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">        0</td><td style="text-align: right;">            1</td><td style="text-align: right;">           4</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        1</td><td style="text-align: right;">           14</td><td style="text-align: right;">          14</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        2</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        3</td><td style="text-align: right;">            2</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        4</td><td style="text-align: right;">            3</td><td style="text-align: right;">          22</td><td style="text-align: right;">          18</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        5</td><td style="text-align: right;">            1</td><td style="text-align: right;">           5</td><td style="text-align: right;">           9</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        6</td><td style="text-align: right;">            5</td><td style="text-align: right;">           9</td><td style="text-align: right;">           6</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        7</td><td style="text-align: right;">            1</td><td style="text-align: right;">          18</td><td style="text-align: right;">          16</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        8</td><td style="text-align: right;">            7</td><td style="text-align: right;">           9</td><td style="text-align: right;">           2</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        9</td><td style="text-align: right;">            1</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       10</td><td style="text-align: right;">           32</td><td style="text-align: right;">          33</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       11</td><td style="text-align: right;">            0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       12</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       13</td><td style="text-align: right;">           18</td><td style="text-align: right;">          23</td><td style="text-align: right;">          13</td><td style="text-align: right;">           6</td></tr>
<tr><td style="text-align: right;">       14</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       15</td><td style="text-align: right;">            0</td><td style="text-align: right;">           6</td><td style="text-align: right;">          10</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       16</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       17</td><td style="text-align: right;">           10</td><td style="text-align: right;">          16</td><td style="text-align: right;">           4</td><td style="text-align: right;">           3</td></tr>
<tr><td style="text-align: right;">       18</td><td style="text-align: right;">            1</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       19</td><td style="text-align: right;">            1</td><td style="text-align: right;">         106</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td></tr>
</tbody>
</table>


## Outlier Detection

Outlier adalah samples janggal yang keluar dari kerumuman. Mereka membuat integritas data tidak sehat.

![](https://paper-attachments.dropbox.com/s_1185AEC62427E23657579AF288686866FF5B3F65A0E36E86D1A293C6B0CCF4B4_1553405161903_sqDCqTEGAmcjqerU4VmkGaw.png)

Suatu sampel $A$ dapat dikatakan sebagai outlier dalam data (D), jika 
$$  \left(\sum^n_{i=1}\left[\operatorname{dist}(A, D_i) > r\right]\right) > \pi{n} $$

dimana $r$ adalah batas normal jarak dan $\pi$ adalah rasio toleransi (antara 0...1). Kedua $r$ dan  $\pi$ dapat diatur secara empiris untuk mendapatkan data yang ideal


```python
r = 20
pi = 0.5
d = df.values

def is_outlier(i):
    count = 0
    n = len(d)
    for j in range(n):
        delta = spad.euclidean(d[i,1:],d[j,1:])
        if (i!=j and delta <= r):
            count += 1
            if count >= pi*n:
                return False
    return True

print("Deteksi outlier dengan r =",r,'dan pi =',pi)
table(pd.DataFrame([[*d[i], 
    'Y' if is_outlier(i) else '-'] for i in range(len(d))], 
    columns=[*df.columns,"Outliers?"]))
```

    Deteksi outlier dengan r = 20 dan pi = 0.5
    


<table>
<thead>
<tr><th style="text-align: right;">  user_id</th><th style="text-align: right;">  pause_video</th><th style="text-align: right;">  play_video</th><th style="text-align: right;">  seek_video</th><th style="text-align: right;">  stop_video</th><th>Outliers?  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">        0</td><td style="text-align: right;">            1</td><td style="text-align: right;">           4</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">        1</td><td style="text-align: right;">           14</td><td style="text-align: right;">          14</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">        2</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td>-          </td></tr>
<tr><td style="text-align: right;">        3</td><td style="text-align: right;">            2</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">        4</td><td style="text-align: right;">            3</td><td style="text-align: right;">          22</td><td style="text-align: right;">          18</td><td style="text-align: right;">           0</td><td>Y          </td></tr>
<tr><td style="text-align: right;">        5</td><td style="text-align: right;">            1</td><td style="text-align: right;">           5</td><td style="text-align: right;">           9</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">        6</td><td style="text-align: right;">            5</td><td style="text-align: right;">           9</td><td style="text-align: right;">           6</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">        7</td><td style="text-align: right;">            1</td><td style="text-align: right;">          18</td><td style="text-align: right;">          16</td><td style="text-align: right;">           0</td><td>Y          </td></tr>
<tr><td style="text-align: right;">        8</td><td style="text-align: right;">            7</td><td style="text-align: right;">           9</td><td style="text-align: right;">           2</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">        9</td><td style="text-align: right;">            1</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td>-          </td></tr>
<tr><td style="text-align: right;">       10</td><td style="text-align: right;">           32</td><td style="text-align: right;">          33</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td><td>Y          </td></tr>
<tr><td style="text-align: right;">       11</td><td style="text-align: right;">            0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">       12</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td>-          </td></tr>
<tr><td style="text-align: right;">       13</td><td style="text-align: right;">           18</td><td style="text-align: right;">          23</td><td style="text-align: right;">          13</td><td style="text-align: right;">           6</td><td>Y          </td></tr>
<tr><td style="text-align: right;">       14</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td>-          </td></tr>
<tr><td style="text-align: right;">       15</td><td style="text-align: right;">            0</td><td style="text-align: right;">           6</td><td style="text-align: right;">          10</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">       16</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td>-          </td></tr>
<tr><td style="text-align: right;">       17</td><td style="text-align: right;">           10</td><td style="text-align: right;">          16</td><td style="text-align: right;">           4</td><td style="text-align: right;">           3</td><td>-          </td></tr>
<tr><td style="text-align: right;">       18</td><td style="text-align: right;">            1</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td>-          </td></tr>
<tr><td style="text-align: right;">       19</td><td style="text-align: right;">            1</td><td style="text-align: right;">         106</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td><td>Y          </td></tr>
</tbody>
</table>


# Outliers Detection 2 

Cara deteksi kedua (lebih efisien) adalah menghitung jarak dari mean setiap fitur ($c$), sehingga sampel $A$ akan menjadi outlier jika

$$ \left(\sum^n_{i=1}\frac{\left(A_c-\overline{c}\right)^2}{\overline{c}}\right) > r $$


```python
# Outliers 2
avgs = [df[x].mean() for x in df.columns][1:]

r = 50
d = df.values

def get_is_outlier(i):
    dist = sum([(c-avgs[j])**2/avgs[j] for j,c in enumerate(d[i,1:])])
    return '{:.2f}'.format(dist), 'Y' if dist > r else '-'

print("Deteksi outlier dengan r =",r)
table(pd.DataFrame([[*d[i], *get_is_outlier(i)] for i in range(len(d))], 
    columns=[*df.columns,"Dist", "Outliers?"]))

```

    Deteksi outlier dengan r = 50
    


<table>
<thead>
<tr><th style="text-align: right;">  user_id</th><th style="text-align: right;">  pause_video</th><th style="text-align: right;">  play_video</th><th style="text-align: right;">  seek_video</th><th style="text-align: right;">  stop_video</th><th style="text-align: right;">  Dist</th><th>Outliers?  </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">        0</td><td style="text-align: right;">            1</td><td style="text-align: right;">           4</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td><td style="text-align: right;"> 12.13</td><td>-          </td></tr>
<tr><td style="text-align: right;">        1</td><td style="text-align: right;">           14</td><td style="text-align: right;">          14</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;"> 21.38</td><td>-          </td></tr>
<tr><td style="text-align: right;">        2</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;"> 23.5 </td><td>-          </td></tr>
<tr><td style="text-align: right;">        3</td><td style="text-align: right;">            2</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;"> 15.62</td><td>-          </td></tr>
<tr><td style="text-align: right;">        4</td><td style="text-align: right;">            3</td><td style="text-align: right;">          22</td><td style="text-align: right;">          18</td><td style="text-align: right;">           0</td><td style="text-align: right;"> 54.1 </td><td>Y          </td></tr>
<tr><td style="text-align: right;">        5</td><td style="text-align: right;">            1</td><td style="text-align: right;">           5</td><td style="text-align: right;">           9</td><td style="text-align: right;">           1</td><td style="text-align: right;"> 14.31</td><td>-          </td></tr>
<tr><td style="text-align: right;">        6</td><td style="text-align: right;">            5</td><td style="text-align: right;">           9</td><td style="text-align: right;">           6</td><td style="text-align: right;">           1</td><td style="text-align: right;">  2.41</td><td>-          </td></tr>
<tr><td style="text-align: right;">        7</td><td style="text-align: right;">            1</td><td style="text-align: right;">          18</td><td style="text-align: right;">          16</td><td style="text-align: right;">           0</td><td style="text-align: right;"> 40.06</td><td>-          </td></tr>
<tr><td style="text-align: right;">        8</td><td style="text-align: right;">            7</td><td style="text-align: right;">           9</td><td style="text-align: right;">           2</td><td style="text-align: right;">           1</td><td style="text-align: right;">  3.56</td><td>-          </td></tr>
<tr><td style="text-align: right;">        9</td><td style="text-align: right;">            1</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;"> 19.78</td><td>-          </td></tr>
<tr><td style="text-align: right;">       10</td><td style="text-align: right;">           32</td><td style="text-align: right;">          33</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td><td style="text-align: right;">182.25</td><td>Y          </td></tr>
<tr><td style="text-align: right;">       11</td><td style="text-align: right;">            0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;"> 20.57</td><td>-          </td></tr>
<tr><td style="text-align: right;">       12</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;"> 23.5 </td><td>-          </td></tr>
<tr><td style="text-align: right;">       13</td><td style="text-align: right;">           18</td><td style="text-align: right;">          23</td><td style="text-align: right;">          13</td><td style="text-align: right;">           6</td><td style="text-align: right;"> 86.56</td><td>Y          </td></tr>
<tr><td style="text-align: right;">       14</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;"> 21.74</td><td>-          </td></tr>
<tr><td style="text-align: right;">       15</td><td style="text-align: right;">            0</td><td style="text-align: right;">           6</td><td style="text-align: right;">          10</td><td style="text-align: right;">           1</td><td style="text-align: right;"> 17.55</td><td>-          </td></tr>
<tr><td style="text-align: right;">       16</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;"> 23.5 </td><td>-          </td></tr>
<tr><td style="text-align: right;">       17</td><td style="text-align: right;">           10</td><td style="text-align: right;">          16</td><td style="text-align: right;">           4</td><td style="text-align: right;">           3</td><td style="text-align: right;">  9.91</td><td>-          </td></tr>
<tr><td style="text-align: right;">       18</td><td style="text-align: right;">            1</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;"> 17   </td><td>-          </td></tr>
<tr><td style="text-align: right;">       19</td><td style="text-align: right;">            1</td><td style="text-align: right;">         106</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td><td style="text-align: right;">636.18</td><td>Y          </td></tr>
</tbody>
</table>


# Handling Missing Values with KNN

KNN (K-Neighboring)



```python
from numpy import nan
from sklearn.impute import KNNImputer
dm = df.values.tolist()
dm[6][2] = nan
dm[9][3] = nan
dfm = pd.DataFrame(dm,columns=df.columns)
print("Before")
table(dfm)

imputer = KNNImputer(n_neighbors=5)
dm = imputer.fit_transform(dm)
dfm = pd.DataFrame(dm,columns=df.columns)
print("After")
table(dfm)
```

    Before
    


<table>
<thead>
<tr><th style="text-align: right;">  user_id</th><th style="text-align: right;">  pause_video</th><th style="text-align: right;">  play_video</th><th style="text-align: right;">  seek_video</th><th style="text-align: right;">  stop_video</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">        0</td><td style="text-align: right;">            1</td><td style="text-align: right;">           4</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        1</td><td style="text-align: right;">           14</td><td style="text-align: right;">          14</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        2</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        3</td><td style="text-align: right;">            2</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        4</td><td style="text-align: right;">            3</td><td style="text-align: right;">          22</td><td style="text-align: right;">          18</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        5</td><td style="text-align: right;">            1</td><td style="text-align: right;">           5</td><td style="text-align: right;">           9</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        6</td><td style="text-align: right;">            5</td><td style="text-align: right;">         nan</td><td style="text-align: right;">           6</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        7</td><td style="text-align: right;">            1</td><td style="text-align: right;">          18</td><td style="text-align: right;">          16</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        8</td><td style="text-align: right;">            7</td><td style="text-align: right;">           9</td><td style="text-align: right;">           2</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        9</td><td style="text-align: right;">            1</td><td style="text-align: right;">           1</td><td style="text-align: right;">         nan</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       10</td><td style="text-align: right;">           32</td><td style="text-align: right;">          33</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       11</td><td style="text-align: right;">            0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       12</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       13</td><td style="text-align: right;">           18</td><td style="text-align: right;">          23</td><td style="text-align: right;">          13</td><td style="text-align: right;">           6</td></tr>
<tr><td style="text-align: right;">       14</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       15</td><td style="text-align: right;">            0</td><td style="text-align: right;">           6</td><td style="text-align: right;">          10</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       16</td><td style="text-align: right;">            0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       17</td><td style="text-align: right;">           10</td><td style="text-align: right;">          16</td><td style="text-align: right;">           4</td><td style="text-align: right;">           3</td></tr>
<tr><td style="text-align: right;">       18</td><td style="text-align: right;">            1</td><td style="text-align: right;">           2</td><td style="text-align: right;">           0</td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       19</td><td style="text-align: right;">            1</td><td style="text-align: right;">         106</td><td style="text-align: right;">           1</td><td style="text-align: right;">           1</td></tr>
</tbody>
</table>


    After
    


<table>
<thead>
<tr><th style="text-align: right;">  user_id</th><th style="text-align: right;">  pause_video</th><th style="text-align: right;">  play_video</th><th style="text-align: right;">  seek_video</th><th style="text-align: right;">  stop_video</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">        0</td><td style="text-align: right;">            1</td><td style="text-align: right;">         4  </td><td style="text-align: right;">         1  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        1</td><td style="text-align: right;">           14</td><td style="text-align: right;">        14  </td><td style="text-align: right;">         0  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        2</td><td style="text-align: right;">            0</td><td style="text-align: right;">         0  </td><td style="text-align: right;">         0  </td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        3</td><td style="text-align: right;">            2</td><td style="text-align: right;">         2  </td><td style="text-align: right;">         0  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        4</td><td style="text-align: right;">            3</td><td style="text-align: right;">        22  </td><td style="text-align: right;">        18  </td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        5</td><td style="text-align: right;">            1</td><td style="text-align: right;">         5  </td><td style="text-align: right;">         9  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        6</td><td style="text-align: right;">            5</td><td style="text-align: right;">         4.2</td><td style="text-align: right;">         6  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        7</td><td style="text-align: right;">            1</td><td style="text-align: right;">        18  </td><td style="text-align: right;">        16  </td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">        8</td><td style="text-align: right;">            7</td><td style="text-align: right;">         9  </td><td style="text-align: right;">         2  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">        9</td><td style="text-align: right;">            1</td><td style="text-align: right;">         1  </td><td style="text-align: right;">         3.2</td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       10</td><td style="text-align: right;">           32</td><td style="text-align: right;">        33  </td><td style="text-align: right;">         1  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       11</td><td style="text-align: right;">            0</td><td style="text-align: right;">         1  </td><td style="text-align: right;">         0  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       12</td><td style="text-align: right;">            0</td><td style="text-align: right;">         0  </td><td style="text-align: right;">         0  </td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       13</td><td style="text-align: right;">           18</td><td style="text-align: right;">        23  </td><td style="text-align: right;">        13  </td><td style="text-align: right;">           6</td></tr>
<tr><td style="text-align: right;">       14</td><td style="text-align: right;">            0</td><td style="text-align: right;">         0  </td><td style="text-align: right;">         1  </td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       15</td><td style="text-align: right;">            0</td><td style="text-align: right;">         6  </td><td style="text-align: right;">        10  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       16</td><td style="text-align: right;">            0</td><td style="text-align: right;">         0  </td><td style="text-align: right;">         0  </td><td style="text-align: right;">           0</td></tr>
<tr><td style="text-align: right;">       17</td><td style="text-align: right;">           10</td><td style="text-align: right;">        16  </td><td style="text-align: right;">         4  </td><td style="text-align: right;">           3</td></tr>
<tr><td style="text-align: right;">       18</td><td style="text-align: right;">            1</td><td style="text-align: right;">         2  </td><td style="text-align: right;">         0  </td><td style="text-align: right;">           1</td></tr>
<tr><td style="text-align: right;">       19</td><td style="text-align: right;">            1</td><td style="text-align: right;">       106  </td><td style="text-align: right;">         1  </td><td style="text-align: right;">           1</td></tr>
</tbody>
</table>



```python

```
