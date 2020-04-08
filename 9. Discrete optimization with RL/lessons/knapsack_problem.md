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



## 🌴 Branch, bound & relaxation
When you do exhaustive search it's basically building a decision tree of 2^n branches. Relaxation methods are to explore the tree without computing all nodes. We iterate two steps: 
- **Branching** (splitting the problem into a number of subproblems like in exhaustive search)
- **Bounding** (finding an optimistic estimate of the best solution to the subproblem, maximization = upper bound & minimization = lower bound)

### How to find an optimization evaluation? How can I relax my problem?
> - We relax a constraint
> - Build the tree and evaluate an optimistic estimate
> - If branching leads to a lower optimisatic estimate, we don't even need to go further in a branch and we can prune it. 

*Branching & bounding can be done a lot of different ways, see Search strategies section*

### What can we relax in the knapsack problem?
- The capacity constraint -> take everything in the knapsack
- The selection variable, we can imagine taking a fraction of each item (xi is now a decimal), this is called **linear relaxation** 

Linear relaxation for the knapsack algorithm works by : 
- Sorting by value density ratio
- Fill the rest of the knapsack with a fraction of the last item that can partially fit, and you have an optimistic estimate for pruning


## 🔍 Search strategies

### Depth-first
Prunes when a node estimation is worse than the best found
- Go deep
- When does it prune? when it finds a new node worse than the found solution
- Is it memory efficient? It can be if we look at a few branches

### Best-first
Select the node with the best estimation
- Go for the best
- When does it prune? when all the nodes are worse than a found solution
- It it memory efficient? If we exaggerate and think of a knapsack with infinite capacity, we will commpute the entire tree, so infinite time and infinite space would be required. When the problem is small, it can be efficient. 


### Least discrepancy or limited discrepancy search
Trust a greedy heuristic
- Assume a good heuristic is available
  - It makes very few mistakes
  - Search tree is binary
  - Following the heuristic means branching left and branching right means the heuristic was wrong
- Limited Discrepancy Search (LDS)
  - Avoid mistakes at all costs
  - Explore the search space in increasing order of mistakes
  - Trusting the heuristic less and less

We explore the search spaces in waves, and trust the heuristic less and less. <br>Its efficiency really depends on a trade off between space and time. 


### And many others search strategies




## 📚 References
- [Wikipedia page on Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem)
