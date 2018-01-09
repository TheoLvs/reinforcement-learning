# Chrome Dino Project
## Playing and solving the Chrome Dinosaur Game with Evolution Strategies and PyTorch
![](http://www.skipser.com/test/trex-game/promotion/trex-chrome-game.png)


##### Summary
- Capturing image from the game - **OK**
- Allowing control programmatically - **OK**
- Trying a simple implementation of rules-based agent with classic CV algorithms - **OK** 
- Capturing scores for fitness and reward - **OK**
- Creating the environment for RL - **OK**
- Developing a RL agent that learns via evolution strategies - **OK**
- Different experiments on both agent and method of learning


##### Ideas 
- Taking as input of the neural network
  - The boundaries of the obstacles in a 1D vector
  - The raw image
  - The processed image
- Initialize the agent with hard coded policy
- Combine the RL agent and the rules-based Agent
- Try other evolution strategies
  - Crossover on the fitness
  - Simple ES
  - CMA-ES


##### Experiments : 
1. **Genetic algorithm** : Generation of 20 dinos, 5 survive, and make 10 offsprings. 10 random dinos are created to complete the 20 population. Did not work at all after 100 generations, still an average score of 50 which is stopping at the first obstacle. This was tested without mutations. The Neural Network is very shallow MLP with one 100-unit hidden layer. 
2. **Genetic algorithm** : Generation of 40 dinos, 10 survive, make 45 offsprings, but 40 are selected at random to recreate the 40-population. Added mutations with gaussian noise at this step. Tried as well with a shallow MLP but also with a simple logistic regression in PyTorch