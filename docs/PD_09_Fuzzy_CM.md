# Fuzzy C-Mean Clustering

Fuzzy adalah clustering menggunakan derajat keanggotaan dengan pendekatan incremental.

## Steps

#### 1. Persiapkan Environment:


```python
from pandas import DataFrame
import random
import numpy as np
from IPython.display import HTML, display
from tabulate import tabulate
from math import log
from sklearn.feature_selection import mutual_info_classif

def table(df): display(HTML(tabulate(df, tablefmt='html', headers='keys', showindex=False)))
```

####  2. Persiapkan Input Data (D = m x n):

  + `n`: Jumlah Sampel
  + `m`: Jumlah Fitur
  + `c`: Jumlah Cluster
  + `w`: Tingkat blur/fuzzy (biasanya 2)
  + `T`: Batas maks Iterasi (biasanya 10)
  + `e`: Akurasi (biasanya 0.1)
  + `Pt`: Fungsi Objektif ke-t
  + `t`: Iterasi ke-t


```python
Data = read_csv('leaf.csv', sep=',')
Data = Data[['Eccentricity','Solidity', 'Lobedness', 'Entropy']].sample(6, random_state=42)
D = Data.values
print("Table (D) >>")
table(D)
```

    Table (D) >>
    


<table>
<thead>
<tr><th style="text-align: right;">      0</th><th style="text-align: right;">      1</th><th style="text-align: right;">        2</th><th style="text-align: right;">      3</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">0.99593</td><td style="text-align: right;">0.80662</td><td style="text-align: right;">2.7342   </td><td style="text-align: right;">0.27303</td></tr>
<tr><td style="text-align: right;">0.50692</td><td style="text-align: right;">0.53024</td><td style="text-align: right;">3.0788   </td><td style="text-align: right;">0.67289</td></tr>
<tr><td style="text-align: right;">0.24465</td><td style="text-align: right;">0.56524</td><td style="text-align: right;">2.854    </td><td style="text-align: right;">0.8331 </td></tr>
<tr><td style="text-align: right;">0.86545</td><td style="text-align: right;">0.82443</td><td style="text-align: right;">0.40204  </td><td style="text-align: right;">1.0136 </td></tr>
<tr><td style="text-align: right;">0.82866</td><td style="text-align: right;">0.9418 </td><td style="text-align: right;">0.11857  </td><td style="text-align: right;">1.8038 </td></tr>
<tr><td style="text-align: right;">0.72719</td><td style="text-align: right;">0.99388</td><td style="text-align: right;">0.0016019</td><td style="text-align: right;">0.9805 </td></tr>
</tbody>
</table>



```python
n, m, c, w, T, e, P0, t = *D.shape, 3, 2, 10, 0.1, 0, 1
print("Variables >>")
print(" n = %d\n m = %d\n c = %d\n w = %d\n T = %d\n e = %f\n P0 = %d\n t = %d" % (n, m, c, w, T, e, P0, t))
```

    Variables >>
     n = 6
     m = 4
     c = 3
     w = 2
     T = 10
     e = 0.100000
     P0 = 0
     t = 1
    

####  3. Siapkan Matrik Derajat Kluster (U = c x n):

Data diisi dengan random atau hasil iterasi lama


```python
random.seed(42)
U = np.array([[random.uniform(0, 1) for _ in range(c)] for _ in range(n)])
print("U >>\n")
print(U)
```

    U >>
    
    [[0.6394268  0.02501076 0.27502932]
     [0.22321074 0.73647121 0.67669949]
     [0.89217957 0.08693883 0.42192182]
     [0.02979722 0.21863797 0.50535529]
     [0.02653597 0.19883765 0.64988444]
     [0.54494148 0.22044062 0.58926568]]
    

#### 4. Hitung Centroid Tiap Cluster (V = m x c):

$$ V_{xy} = \frac{\sum^n_{i=1}(U_{iy})^w\times{D_{ix}}}{\sum^n_{i=1}(U_{iy})^w} $$


```python
# Caution: NP Array is math-agnostic (column-by-column)
def cluster(U, D, x, y): return sum([U[i,y]**w*D[i,x] for i in range(n)])/sum([U[i,y]**w for i in range(n)])
V = np.array([[cluster(U,D,x,y) for x in range(m)] for y in range(c)])
print("V >>\n")
print(V)
```

    V >>
    
    [[0.54370379 0.70992788 2.28168401 0.7092545 ]
     [0.56356398 0.60788268 2.50132412 0.78491767]
     [0.67635682 0.7819355  1.31182068 1.05856209]]
    

#### 5. Hitung Fungsi Objektif pada t (Pt)

$$ P_t = \sum^n_{i=1}\sum^c_{k=1}\left(\left[\sum^m_{j=1}\left(D_{ij}-V_{kj}\right)^2\right](U_{ik})^w\right) $$


```python
def objective(V,U,D): return sum([sum([sum([(D[i,j]-V[k,j])**2 for j in range(m)])*(U[i,k]**w) for k in range(c)]) for i in range(n)])
Pt = objective(V,U,D)
print("Pt >>\n")
print(Pt)
```

    Pt >>
    
    7.165764247017886
    

#### 6. Hitung Ulang Matrik Derajat Kluster (U = c x n):

$$ U_{ik} = \frac{\left[\sum^m_{j=1}(D_{ij}-V_{kj})^2\right]^{\frac{-1}{w-1}}}{\sum^c_{k=1}\left[\sum^m_{j=1}(D_{ij}-V_{kj})^2\right]^{\frac{-1}{w-1}}} $$


```python
def converge(V,D,i,k): return (sum([(D[i,j]-V[k,j])**2 for j in range(m)])**(-1/(w-1)))/sum([sum([(D[i,j]-V[k,j])**2 for j in range(m)])**(-1/(w-1)) for k in range(c)])
print("U >>\n")
np.array([[converge(V,D,i,k) for k in range(c)] for i in range(n)])
```




    array([[0.42661745, 0.47867606, 0.09470648],
           [0.32401778, 0.61139512, 0.0645871 ],
           [0.31857727, 0.62718924, 0.05423349],
           [0.16315857, 0.13281473, 0.7040267 ],
           [0.20677417, 0.18023246, 0.61299337],
           [0.20507176, 0.17092863, 0.62399961]])



#### 7. Cek Berhenti Atau Loop Kembali

Jika $ P_t - P_{t-1} < e $ atau $ t >= T $ maka **BERHENTI**

Jika tidak, ulangi langkah dari **Hitung Centroid Tiap Cluster** 


```python
def iterate(U):
    V = np.array([[cluster(U, D, x, y) for x in range(m)] for y in range(c)])
    return np.array([[converge(V,D,i,k) for k in range(c)] for i in range(n)]), objective(V,U,D)

def fuzzyCM(U):
    #U = np.array([[random.uniform(0, 1) for _ in range(c)] for _ in range(n)])
    
    U, P2, P, t = *iterate(U), 0, 1
    while abs(P2 - P) > e and t < T:
        U, P2, P, t = *iterate(U), P2, t+1
    return U, t

FuzzyResult, FuzzyIters = fuzzyCM(U)
print("Iterating %d times, fuzzy result >> \n" % FuzzyIters)
print(FuzzyResult)
```

    Iterating 5 times, fuzzy result >> 
    
    [[9.99946808e-01 4.84760420e-05 4.71615530e-06]
     [5.97625571e-02 9.36333265e-01 3.90417747e-03]
     [3.61938911e-02 9.59438085e-01 4.36802436e-03]
     [1.97778963e-02 1.70437266e-02 9.63178377e-01]
     [3.11702255e-02 3.00196318e-02 9.38810143e-01]
     [1.40843238e-02 1.23997912e-02 9.73515885e-01]]
    

#### 8. Ambil Nilai Terbesar pada Kolom Sebagai Cluster pada setiap Record Data


```python
table(DataFrame([D[i].tolist()+[np.argmax(FuzzyResult[i].tolist())] for i in range(n)], columns=Data.columns.tolist()+["Cluster Index"]))
```


<table>
<thead>
<tr><th style="text-align: right;">  Eccentricity</th><th style="text-align: right;">  Solidity</th><th style="text-align: right;">  Lobedness</th><th style="text-align: right;">  Entropy</th><th style="text-align: right;">  Cluster Index</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">       0.99593</td><td style="text-align: right;">   0.80662</td><td style="text-align: right;">  2.7342   </td><td style="text-align: right;">  0.27303</td><td style="text-align: right;">              0</td></tr>
<tr><td style="text-align: right;">       0.50692</td><td style="text-align: right;">   0.53024</td><td style="text-align: right;">  3.0788   </td><td style="text-align: right;">  0.67289</td><td style="text-align: right;">              1</td></tr>
<tr><td style="text-align: right;">       0.24465</td><td style="text-align: right;">   0.56524</td><td style="text-align: right;">  2.854    </td><td style="text-align: right;">  0.8331 </td><td style="text-align: right;">              1</td></tr>
<tr><td style="text-align: right;">       0.86545</td><td style="text-align: right;">   0.82443</td><td style="text-align: right;">  0.40204  </td><td style="text-align: right;">  1.0136 </td><td style="text-align: right;">              2</td></tr>
<tr><td style="text-align: right;">       0.82866</td><td style="text-align: right;">   0.9418 </td><td style="text-align: right;">  0.11857  </td><td style="text-align: right;">  1.8038 </td><td style="text-align: right;">              2</td></tr>
<tr><td style="text-align: right;">       0.72719</td><td style="text-align: right;">   0.99388</td><td style="text-align: right;">  0.0016019</td><td style="text-align: right;">  0.9805 </td><td style="text-align: right;">              2</td></tr>
</tbody>
</table>

