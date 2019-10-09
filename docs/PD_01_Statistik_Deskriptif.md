# Statistik Deskriptif

Setiap kelompok data numerik mempunyai properti tersendiri yang menjelaskan secara unik data tersebut:

Sebelum mulai, mari kita mengambil beberapa sampel data


```python
import pandas as pd
import statistics, itertools
from IPython.display import HTML, display
from tabulate import tabulate

def table(df): display(HTML(tabulate(df, tablefmt='html', headers='keys', showindex=False)))
```


```python
df = pd.read_csv('leaf.csv', nrows=9, usecols=['Aspect Ratio']) # Ambil Sampel
data = [round(x[0],1) for x in df.values] # Bulat-bulat
df = pd.DataFrame(data, columns=['sample']); # Data Frame dari Sample
dc = df['sample'] # Data set kolom sample
dc
```




    0    1.5
    1    1.5
    2    1.6
    3    1.5
    4    1.8
    5    1.5
    6    1.8
    7    1.6
    8    1.8
    Name: sample, dtype: float64



Data ini dapat kita deskripsikan menggunakan properti-properti berikut.

## Mean

Mean adalah rata-rata dari suatu dataset. Diperoleh dari sum dataset lalu dibagi dengan jumlah elemen dataset. Biasa disimbolkan sebagai $\mu$

$$ \overline{x}=\frac{\sum_{i=1}^{N} x_{i}}{N}=\frac{x_{1}+x_{2}+\cdots+x_{N}}{N} $$


```python
print("Rata-rata", dc.values, "=", dc.mean())
```

    Rata-rata [1.5 1.5 1.6 1.5 1.8 1.5 1.8 1.6 1.8] = 1.6222222222222222
    

Mean dalam built-in python:


```python
print("Rata-rata", data, "=", statistics.mean(data))
```

    Rata-rata [1.5, 1.5, 1.6, 1.5, 1.8, 1.5, 1.8, 1.6, 1.8] = 1.6222222222222222
    

## Median

Median merupakan titik data yang paling baik apabila dataset telah diurutkan. Dalam data numerik non interval, data ke-(n-1)/2 adalah median jika n ganjil atau rata-rata dari data ke-(n/2) dan data ke-(n/2+1) jika n genap.


```python
print("Median", dc.values, "=", dc.median())
```

    Median [1.5 1.5 1.6 1.5 1.8 1.5 1.8 1.6 1.8] = 1.6
    

Median dalam built-in python:


```python
print("Median", data, "=", statistics.median(data))
```

    Median [1.5, 1.5, 1.6, 1.5, 1.8, 1.5, 1.8, 1.6, 1.8] = 1.6
    

Pembuktian dengan menyortir + eliminasi data:


```python
sorteddata = data[:]; sorteddata.sort(); 
print(sorteddata)
while(len(sorteddata)>1):
    sorteddata = sorteddata[1:-1]
    print(sorteddata)
```

    [1.5, 1.5, 1.5, 1.5, 1.6, 1.6, 1.8, 1.8, 1.8]
    [1.5, 1.5, 1.5, 1.6, 1.6, 1.8, 1.8]
    [1.5, 1.5, 1.6, 1.6, 1.8]
    [1.5, 1.6, 1.6]
    [1.6]
    

## Mode

Mode merupakan statistik untuk angka mana yang paling banyak frekuensinya dalam dataset. Mode bisa dalam bentuk diskrit atau kelompok.

Mode (diskrik) dalam built-in python:


```python
print("Median", data, "=", statistics.mode(data))
```

    Median [1.5, 1.5, 1.6, 1.5, 1.8, 1.5, 1.8, 1.6, 1.8] = 1.5
    

`scipy` mempunyai tool untuk mendeteksi mode secara lebih detail jika ada >1 value dengan frekuensi yang sama


```python
from scipy import stats
from numpy import transpose
modedata = stats.mode(dc)
table(pd.DataFrame(transpose([modedata.mode, modedata.count]), columns=["Mode", "Count"]))
```


<table>
<thead>
<tr><th style="text-align: right;">  Mode</th><th style="text-align: right;">  Count</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">   1.5</td><td style="text-align: right;">      4</td></tr>
</tbody>
</table>


Gunakan `seaborn` untuk melihat frekuensi secara grafikal:


```python
from seaborn import distplot
ax = distplot(data)
```

## Range
Range dalam suatu dataset ialah angka tertinggi dan angka terendah dalam dataset.


```python
print("Range", dc.values, ", Max:", dc.max(), "Min:", dc.min())
```

    Range [1.5 1.5 1.6 1.5 1.8 1.5 1.8 1.6 1.8] , Max: 1.8 Min: 1.5
    

Range dalam built-in python:


```python
print("Range", data, ", Max:", max(data), "Min:", min(data))
```

    Range [1.5, 1.5, 1.6, 1.5, 1.8, 1.5, 1.8, 1.6, 1.8] , Max: 1.8 Min: 1.5
    

## Quantile

Quantil adalah jarak data yang memisahkan data sekin persen dari yang terkecil hingga tertinggi. Quantil dipisah menjadi:
+ Q1 sebagai Quantil bawah (25%) 
+ Q2 sebagai Quantil tengah (50%)
+ Q3 sebagai Quantil atas (75%)


```python
print("Quantil", dc.values, "Q1:", dc.quantile(0.25), "Q2:", dc.quantile(0.5), "Q3:", dc.quantile(0.75))
```

    Quantil [1.5 1.5 1.6 1.5 1.8 1.5 1.8 1.6 1.8] Q1: 1.5 Q2: 1.6 Q3: 1.8
    

Quantil bisa dihitung menggunakan `numpy`:


```python
from numpy import quantile
print("Quantil", data, "Q1:", quantile(data, 0.25), "Q2:", dc.quantile(0.5), "Q3:", dc.quantile(0.75))
```

    Quantil [1.5, 1.5, 1.6, 1.5, 1.8, 1.5, 1.8, 1.6, 1.8] Q1: 1.5 Q2: 1.6 Q3: 1.8
    

## Variance

Properti tentang seberapa jauh nilai dari rata-rata (alias variasi). Biasa disimbolkan $\sigma^2$


```python
print("Variansi", dc.values, "=", dc.var())
```

    Variansi [1.5 1.5 1.6 1.5 1.8 1.5 1.8 1.6 1.8] = 0.01944444444444445
    


```python
print("Variansi", data, "=", statistics.variance(data))
```

    Variansi [1.5, 1.5, 1.6, 1.5, 1.8, 1.5, 1.8, 1.6, 1.8] = 0.01944444444444445
    

## Standar Deviasi

Standar nilai tentang seberapa jauh data dari mean. Rumus:

$$ \sqrt{\frac{\sum_{i=1}^n\left(x_i-\overline{x}\right)^2}{n-1}} $$


```python
print("Standar Deviasi", dc.values, "=", dc.std())
```

    Standar Deviasi [1.5 1.5 1.6 1.5 1.8 1.5 1.8 1.6 1.8] = 0.1394433377556793
    


```python
print("Variansi", data, "=", statistics.stdev(data))
```

    Variansi [1.5, 1.5, 1.6, 1.5, 1.8, 1.5, 1.8, 1.6, 1.8] = 0.1394433377556793
    

## Summary

Deskripsi Data set dalam `leaf.csv`:


```python
adf = pd.read_csv('leaf.csv')
adata = [[x, 
    '{:.2f}'.format(adf[x].mean()), 
    '{:.2f}'.format(adf[x].median()), 
    '{:.2f}'.format(adf[x].min()), 
    '{:.2f}'.format(adf[x].max()),
    '{:.2f}'.format(adf[x].skew()),
    '{:.2f}'.format(adf[x].var()),
    '{:.2f}'.format(adf[x].std())] 
        for x in adf.columns[1:]]

table(pd.DataFrame(adata, columns=['Nama Kolom', 'Mean', 'Median', 'Min', 'Max', 'Skew', 'Var', 'Std']))
```


<table>
<thead>
<tr><th>Nama Kolom               </th><th style="text-align: right;">  Mean</th><th style="text-align: right;">  Median</th><th style="text-align: right;">  Min</th><th style="text-align: right;">  Max</th><th style="text-align: right;">  Skew</th><th style="text-align: right;">  Var</th><th style="text-align: right;">  Std</th></tr>
</thead>
<tbody>
<tr><td>Specimen Number          </td><td style="text-align: right;">  6.28</td><td style="text-align: right;">    6   </td><td style="text-align: right;"> 1   </td><td style="text-align: right;">16   </td><td style="text-align: right;">  0.2 </td><td style="text-align: right;">11.99</td><td style="text-align: right;"> 3.46</td></tr>
<tr><td>Eccentricity             </td><td style="text-align: right;">  0.72</td><td style="text-align: right;">    0.76</td><td style="text-align: right;"> 0.12</td><td style="text-align: right;"> 1   </td><td style="text-align: right;"> -0.56</td><td style="text-align: right;"> 0.04</td><td style="text-align: right;"> 0.21</td></tr>
<tr><td>Aspect Ratio             </td><td style="text-align: right;">  2.44</td><td style="text-align: right;">    1.57</td><td style="text-align: right;"> 1.01</td><td style="text-align: right;">19.04</td><td style="text-align: right;">  3.33</td><td style="text-align: right;"> 6.76</td><td style="text-align: right;"> 2.6 </td></tr>
<tr><td>Elongation               </td><td style="text-align: right;">  0.51</td><td style="text-align: right;">    0.5 </td><td style="text-align: right;"> 0.11</td><td style="text-align: right;"> 0.95</td><td style="text-align: right;">  0.34</td><td style="text-align: right;"> 0.04</td><td style="text-align: right;"> 0.2 </td></tr>
<tr><td>Solidity                 </td><td style="text-align: right;">  0.9 </td><td style="text-align: right;">    0.95</td><td style="text-align: right;"> 0.49</td><td style="text-align: right;"> 0.99</td><td style="text-align: right;"> -2.06</td><td style="text-align: right;"> 0.01</td><td style="text-align: right;"> 0.11</td></tr>
<tr><td>Stochastic Convexity     </td><td style="text-align: right;">  0.94</td><td style="text-align: right;">    0.99</td><td style="text-align: right;"> 0.4 </td><td style="text-align: right;"> 1   </td><td style="text-align: right;"> -2.63</td><td style="text-align: right;"> 0.01</td><td style="text-align: right;"> 0.12</td></tr>
<tr><td>Isoperimetric Factor     </td><td style="text-align: right;">  0.53</td><td style="text-align: right;">    0.58</td><td style="text-align: right;"> 0.08</td><td style="text-align: right;"> 0.86</td><td style="text-align: right;"> -0.48</td><td style="text-align: right;"> 0.05</td><td style="text-align: right;"> 0.22</td></tr>
<tr><td>Maximal Indentation Depth</td><td style="text-align: right;">  0.04</td><td style="text-align: right;">    0.02</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.2 </td><td style="text-align: right;">  1.71</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.04</td></tr>
<tr><td>Lobedness                </td><td style="text-align: right;">  0.52</td><td style="text-align: right;">    0.1 </td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 7.21</td><td style="text-align: right;">  3.12</td><td style="text-align: right;"> 1.08</td><td style="text-align: right;"> 1.04</td></tr>
<tr><td>Average Intensity        </td><td style="text-align: right;">  0.05</td><td style="text-align: right;">    0.04</td><td style="text-align: right;"> 0.01</td><td style="text-align: right;"> 0.19</td><td style="text-align: right;">  0.94</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.04</td></tr>
<tr><td>Average Contrast         </td><td style="text-align: right;">  0.12</td><td style="text-align: right;">    0.12</td><td style="text-align: right;"> 0.03</td><td style="text-align: right;"> 0.28</td><td style="text-align: right;">  0.47</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.05</td></tr>
<tr><td>Smoothness               </td><td style="text-align: right;">  0.02</td><td style="text-align: right;">    0.01</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.07</td><td style="text-align: right;">  1.21</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.01</td></tr>
<tr><td>Third moment             </td><td style="text-align: right;">  0.01</td><td style="text-align: right;">    0   </td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.03</td><td style="text-align: right;">  1.78</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0.01</td></tr>
<tr><td>Uniformity               </td><td style="text-align: right;">  0   </td><td style="text-align: right;">    0   </td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0   </td><td style="text-align: right;">  2.13</td><td style="text-align: right;"> 0   </td><td style="text-align: right;"> 0   </td></tr>
<tr><td>Entropy                  </td><td style="text-align: right;">  1.16</td><td style="text-align: right;">    1.08</td><td style="text-align: right;"> 0.17</td><td style="text-align: right;"> 2.71</td><td style="text-align: right;">  0.49</td><td style="text-align: right;"> 0.34</td><td style="text-align: right;"> 0.58</td></tr>
</tbody>
</table>



```python

```
