# Traveling Salesman Problem

Optimization attempts


```python
from numpy import transpose, array
from matplotlib.pyplot import scatter, subplots
from scipy.spatial import distance_matrix
from pandas import DataFrame

points = array([(0,0), (0,4), (3,0), (1,1), (5,5)])
labels = ['A', 'B', 'C', 'D', 'E']
x, y = transpose(points)

fig, ax = subplots()
ax.scatter(x, y)

for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i], y[i]))

df = DataFrame(distance_matrix(points, points), columns=labels)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.414214</td>
      <td>7.071068</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>3.162278</td>
      <td>5.099020</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>2.236068</td>
      <td>5.385165</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.414214</td>
      <td>3.162278</td>
      <td>2.236068</td>
      <td>0.000000</td>
      <td>5.656854</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7.071068</td>
      <td>5.099020</td>
      <td>5.385165</td>
      <td>5.656854</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import math
import itertools

def isRouteValid(routes):
    counts = {}
    for f, t, d in routes:
        f = int(f); t = int(t)
        if (f == t): return False
        counts[f] = 1 if counts.get(f, None) is None else counts[f] + 1
        counts[t] = 1 if counts.get(t, None) is None else counts[t] + 1
    for c in counts:
        if (counts[c] != 2): return False
    return True

def mapEnroute(dists, route):
    r = []
    for i in range(len(route)):
        r.append(dists[i][route[i]])
    return r

def nicePrint(pds, i):
    m = [pds[x][y] for x,y in enumerate(i)]
    label = lambda x: labels[int(x)]
    s = sum([d for f,t,d in m])
    return str(["{0} -> {1} {2:.2f}".format(label(f),label(t),d) for f,t,d in m]) + " {:.2f}".format(s)

def cost(distances, labels):
    dists = []
    for x, row in enumerate(distances):
        for y, cell in enumerate(row):
            dists.append([x, y, cell])
    pd = DataFrame(dists, columns=['f','t','d'])
    pds = [list(data.sort_values('d').values) for key,data in pd.groupby('t')]
    
    iters = list(itertools.product(*([list(range(len(labels)))]*len(labels))))
    iters.sort(key=lambda x: sum(x))
    for i in (iters):
        if(isRouteValid(mapEnroute(pds, i))):
            return nicePrint(pds, i)
    return 0

def bruteIt(distances, labels):
    dists = []
    for x, row in enumerate(distances):
        for y, cell in enumerate(row):
            dists.append([x, y, cell])
    pd = DataFrame(dists, columns=['f','t','d'])
    pds = [list(data.sort_values('d').values) for key,data in pd.groupby('t')]
    
    iters = list(itertools.permutations(list(range(len(labels)))))
    shortest_route = None
    shortest_dist = 999999999999999999
    for i in iters:
        if any([x==y for x,y in enumerate(i)]): continue
        distance = sum([distances[x,y] for x,y in enumerate(i)])
        if distance < shortest_dist:
            shortest_dist = distance
            shortest_route = i
    return shortest_dist, ["{}{} {:.2f}".format(labels[y],labels[x],distances[x,y]) for x,y in enumerate(shortest_route)], shortest_dist

from timeit import default_timer as timer
start = timer()
print(cost(df.values, labels))
end = timer()
print(end - start,"s")
start = timer()
print(bruteIt(df.values, labels))
end = timer()
print(end - start,"s")
```

    ['C -> A 3.00', 'D -> B 3.16', 'E -> C 5.39', 'A -> D 1.41', 'B -> E 5.10'] 18.06
    0.06618459999981496 s
    (16.84832056705845, ['DA 1.41', 'EB 5.10', 'AC 3.00', 'CD 2.24', 'BE 5.10'], 16.84832056705845)
    0.011839300000019648 s
    

> All code in GPL
