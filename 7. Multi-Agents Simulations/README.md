# Multi-Agents simulation
![](https://thumbs.gfycat.com/EvergreenGenuineAmethystgemclam-size_restricted.gif)

Simulations including multiple agents are present everywhere in our daily lives, from large-scale economics policies to epidemiology. <br>
Agent-based modeling is even more effective when merged with modern AI techniques such as Reinforcement Learning. <br>
This folder contains experiments on this topics

# Experiments summary
- **October 2019** - First attempts to create a Sugarscape experiment. Developed a framework using Dataframes for accelerated computations. Yet too many interactions to code from scratch and low performance
- **December 2019** - Discovered Unity for such simulations + ML Agents
- **March 2019** - Due to COVID19 outbreak, I started experiments on Multi Agent modeling and social distancing. PyGame is a good candidate for 2D simulations similar to Unity but in Python. Many possibilities and spatial O(n2) interactions are really sped up thanks to colliders embedded in PyGame. Movements are still feasible up to 10k agents at least in my first experiments.  


# References
## Libraries & softwares
- Unity
- NetLogo
- [MESA](https://github.com/projectmesa/mesa) - Python
- [SPADE](https://spade-mas.readthedocs.io/en/latest/readme.html) - Python
- [abcEconomics](https://abce.readthedocs.io/en/master/)
- [GAMA-Platform](https://gama-platform.github.io/)
- [Manim](https://github.com/3b1b/manim) by the great Grant Sanderson
- PyGame

## Tutorials
- [Introduction to Agent Based Modeling in Python](https://towardsdatascience.com/introduction-to-mesa-agent-based-modeling-in-python-bcb0596e1c9a)

## Inspiration
- https://www.complexity-explorables.org/
- Sugarscape https://www.youtube.com/watch?v=r_It_X7v-1E

## Features to implement
- Set and reload data -> ok
- Animation over the simulation (gif ok, ipywidgets to go)
- Action framework with delayed deferrence
- Metrics storage for each agent
- Set up geographical zones and 2D maps with impossible moves
- Find closest agent method
- Wander method
- Launch simulation until certain time + early stopping