# Seleksi Fitur

Kita dapat menghitung "seberapa berharga" fitur X dalam data melalui Feature Gain. Dengan demikian, fitur terlalu banyak bisa dikurangi.


```python
from pandas import *
from IPython.display import HTML, display
from tabulate import tabulate
from math import log
from sklearn.feature_selection import mutual_info_classif

def table(df): display(HTML(tabulate(df, tablefmt='html', headers='keys', showindex=False)))
```

Mari kita ambil beberapa sampel:


```python
df = read_csv('play.csv', sep=';')
table(df)
```


<table>
<thead>
<tr><th>outlook  </th><th>temperature  </th><th>humidity  </th><th>windy  </th><th>play  </th></tr>
</thead>
<tbody>
<tr><td>sunny    </td><td>hot          </td><td>high      </td><td>False  </td><td>no    </td></tr>
<tr><td>sunny    </td><td>hot          </td><td>high      </td><td>True   </td><td>no    </td></tr>
<tr><td>overcast </td><td>hot          </td><td>high      </td><td>False  </td><td>yes   </td></tr>
<tr><td>rainy    </td><td>mild         </td><td>high      </td><td>False  </td><td>yes   </td></tr>
<tr><td>rainy    </td><td>cool         </td><td>normal    </td><td>False  </td><td>yes   </td></tr>
<tr><td>rainy    </td><td>cool         </td><td>normal    </td><td>True   </td><td>no    </td></tr>
<tr><td>overcast </td><td>cool         </td><td>normal    </td><td>True   </td><td>yes   </td></tr>
<tr><td>sunny    </td><td>mild         </td><td>high      </td><td>False  </td><td>no    </td></tr>
<tr><td>sunny    </td><td>cool         </td><td>normal    </td><td>False  </td><td>yes   </td></tr>
<tr><td>rainy    </td><td>mild         </td><td>normal    </td><td>False  </td><td>yes   </td></tr>
<tr><td>sunny    </td><td>mild         </td><td>normal    </td><td>True   </td><td>yes   </td></tr>
<tr><td>overcast </td><td>mild         </td><td>high      </td><td>True   </td><td>yes   </td></tr>
<tr><td>overcast </td><td>hot          </td><td>normal    </td><td>False  </td><td>yes   </td></tr>
<tr><td>rainy    </td><td>mild         </td><td>high      </td><td>True   </td><td>no    </td></tr>
</tbody>
</table>


## Entropy Target

Entropy (keberagaman) kolom target:

$$ E(T) = \sum_{i=1}^n {-P_i\log{P_i}} $$


dimana $P$ = Rasio Peluang muncul dalam record


```python
def findEntropy(column):
    rawGroups = df.groupby(column)
    targetGroups = [[key, len(data), len(data)/df[column].size] for key,data in rawGroups]
    targetGroups = DataFrame(targetGroups, columns=['value', 'count', 'probability'])
    return sum([-x*log(x,2) for x in targetGroups['probability']]), targetGroups, rawGroups

entropyTarget, groupTargets, _ = findEntropy('play')
table(groupTargets)
print('entropy target =', entropyTarget)
```


<table>
<thead>
<tr><th>value  </th><th style="text-align: right;">  count</th><th style="text-align: right;">  probability</th></tr>
</thead>
<tbody>
<tr><td>no     </td><td style="text-align: right;">      5</td><td style="text-align: right;">     0.357143</td></tr>
<tr><td>yes    </td><td style="text-align: right;">      9</td><td style="text-align: right;">     0.642857</td></tr>
</tbody>
</table>


    entropy target = 0.9402859586706309
    

## Gain

Gain dalam sebuah fitur $X$ untuk data $T$:

$$ \operatorname{Gain}(T, X) = \operatorname{Entropy}(T) - \sum_{v\in{T}} \frac{T_{X,v}}{T} E(T_{X,v}) $$





```python
def findGain(column):
    entropyOutlook, groupOutlooks, rawOutlooks = findEntropy(column)
    table(groupOutlooks)
    gain = entropyTarget-sum(len(data)/len(df)*sum(-x/len(data)*log(x/len(data),2) 
                for x in data.groupby('play').size()) for key,data in rawOutlooks)
    print("gain dari '%s': %f" % (column, gain))
    return gain

gains = [[x,findGain(x)] for x in ['outlook','temperature','humidity','windy']]
```


<table>
<thead>
<tr><th>value   </th><th style="text-align: right;">  count</th><th style="text-align: right;">  probability</th></tr>
</thead>
<tbody>
<tr><td>overcast</td><td style="text-align: right;">      4</td><td style="text-align: right;">     0.285714</td></tr>
<tr><td>rainy   </td><td style="text-align: right;">      5</td><td style="text-align: right;">     0.357143</td></tr>
<tr><td>sunny   </td><td style="text-align: right;">      5</td><td style="text-align: right;">     0.357143</td></tr>
</tbody>
</table>


    gain dari 'outlook': 0.246750
    


<table>
<thead>
<tr><th>value  </th><th style="text-align: right;">  count</th><th style="text-align: right;">  probability</th></tr>
</thead>
<tbody>
<tr><td>cool   </td><td style="text-align: right;">      4</td><td style="text-align: right;">     0.285714</td></tr>
<tr><td>hot    </td><td style="text-align: right;">      4</td><td style="text-align: right;">     0.285714</td></tr>
<tr><td>mild   </td><td style="text-align: right;">      6</td><td style="text-align: right;">     0.428571</td></tr>
</tbody>
</table>


    gain dari 'temperature': 0.029223
    


<table>
<thead>
<tr><th>value  </th><th style="text-align: right;">  count</th><th style="text-align: right;">  probability</th></tr>
</thead>
<tbody>
<tr><td>high   </td><td style="text-align: right;">      7</td><td style="text-align: right;">          0.5</td></tr>
<tr><td>normal </td><td style="text-align: right;">      7</td><td style="text-align: right;">          0.5</td></tr>
</tbody>
</table>


    gain dari 'humidity': 0.151836
    


<table>
<thead>
<tr><th>value  </th><th style="text-align: right;">  count</th><th style="text-align: right;">  probability</th></tr>
</thead>
<tbody>
<tr><td>False  </td><td style="text-align: right;">      8</td><td style="text-align: right;">     0.571429</td></tr>
<tr><td>True   </td><td style="text-align: right;">      6</td><td style="text-align: right;">     0.428571</td></tr>
</tbody>
</table>


    gain dari 'windy': 0.048127
    

### Overall Gain Score:


```python
result = DataFrame(gains, columns=["Feature", "Gain Score"]).sort_values("Gain Score")[::-1]
table(result)

print("'%s' mempunyai gain score tertinggi sedangkan '%s' terendah" % (result.values[0,0], result.values[-1,0]))
```


<table>
<thead>
<tr><th>Feature    </th><th style="text-align: right;">  Gain Score</th></tr>
</thead>
<tbody>
<tr><td>outlook    </td><td style="text-align: right;">   0.24675  </td></tr>
<tr><td>humidity   </td><td style="text-align: right;">   0.151836 </td></tr>
<tr><td>windy      </td><td style="text-align: right;">   0.048127 </td></tr>
<tr><td>temperature</td><td style="text-align: right;">   0.0292226</td></tr>
</tbody>
</table>


    'outlook' mempunyai gain score tertinggi sedangkan 'temperature' terendah
    
