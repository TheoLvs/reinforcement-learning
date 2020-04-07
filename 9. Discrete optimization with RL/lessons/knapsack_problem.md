# Knapsack Problem
![](https://miro.medium.com/max/684/0*3dS6Jw8NzzSD-mn8.jpg)


## 📝 Conventions & notations
- I = {1,2,...,n}
- O(k,j) denotes an optimal solution to the knapsack problem with capacity k and items [1,...,j]. This is what we want to solve. 

## 👜 Modeling the Knapsack Problem

### Defining the problem
- **Variables**
  - Decision variables
    - ``xi`` denotes whether the item i is selected in the solution
  - Other variables
    - ``wi`` denotes the weight of the item i
    - ``vi`` denotes the value of the item i
- **Problem constraint**
  - Selected item cannot exceed the capacity of the backpack ``sum(wi*xi) <= K`` 
- **Objective function**
  - We want to maximize ``sum(vi*xi)``


### Number of configurations
- How many possible configurations of 1 and 0 for ``(x1,x2,...,xn)`` ? -> Search space
- Not all of them are feasible -> Feasible search space
- How many are they ? ``2^n`` -> exponential growth -> brute force is not possible for more than a few objects


## 🤗 Greedy algorithms

### Greedy algorithms to solve the knapsack problem
1. Take lighter item first
2. Take most valuable item first
3. Compute value density ratio (value/weight) and take the most important value

For one problem, **there are many greedy algorithms**. With no guarantee it's optimal. It really depends on the input. But it's quick to implement, it's often fast to run and it serves as a baseline. 

### Advantages
- Quick to design and implement
- Can be very fast

### Problems
- No quality guarantee
- Quality can vary widely on the input
- Problem feasibility needs to be easy




## ⚡ Dynamic Programming
### Recurrence relations (Bellmann equations)
We want to solve O(k,j) by recurrence : 
- Assume we know how to solve ``O(k,j-1)`` for all k, and we want to solve ``O(k,j)`` by adding one more item : the item ``j``
- If ``wj <= k`` there are two cases: 
  - Either we don't select item j and the best solution is then ``O(k,j-1)``
  - Or we select item j and the best solution is ``vj + O(k-wj,j-1)``
- Or written mathematically 
```
- O(k,j) = max(O(k,j-1),vj + O(k-wj,j-1)) if wj <=k
- O(k,j) = O(k,j-1) otherwise
```
- And of course ``O(k,0) = 0`` for all k (there are no items, there is no value)

### Recursive function in Python 
```python
# Variables
w = list(...)
v = list(...)

def O(k,j):
    if (j == 0):
        return 0
    elif w[j] <= k:
        return max([O(k,j-1),v[j] + O(k-w[j],j-1)])
    else:
        return O(k,j-1)      
```
How efficient is this approach? Not a lot if we go top down (to compute many values we need to compute again the same values, that's often the case with complex recursive functions). <br>
That's why Dynamic Programming is all about Bottom-up approach. ###

### Bottom-up computation
- Compute the recursive equations bottom up
  - Start with zero items
  - Add one more item, then two ... 

Often needs to be thought as a tables (capacity x items)

![](https://sadakurapati.files.wordpress.com/2013/11/knapsack2.png?w=584)

- Building the table one by one using the formula
- Tracing back to find the optimal solution

### Efficiency
- Complexity of the algorithm -> time to fill the table ie O(Kn), we could think it's polynomial not exactly
- It's not polynomial, but exponential because K is represented in a computer by log(K) bits. So we call this type of algorithms pseudo-polynomials. Because it's only efficient when K is small


## 📚 References
- [Wikipedia page on Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem)
