# SpaceLanding
Capsule Landing - DRL Project (DQN)
![capsule_lander](https://user-images.githubusercontent.com/85990934/137446158-eb997306-5a20-47bb-9ef3-b586f247d08d.png)

## Goals 
* Learn how to build environment
* Learn how to build agent
* Learn how agent should interact with environment
* Practice, and see the whole picture to get better understanding of the system 


## Dependencies
pytorch - https://pytorch.org/get-started/locally/
```sh
pip install numpy
pip install matplotlib
pip install pygame
```

## About the project
Target: 
Fast reach and land the capsule on the ground

Reward system:
First, we should reach the landing space as fast as possible, so we reward high falling speed which can be achieved by rotating the nose of the capsule down and turning on main engine.
Second, we should land correctly, so we reward rotating capsule to reach 0 position (when nose looks up) and a smooth landing 

## Resources
Grokking DRL - https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_09/chapter-09.ipynb

Gym CartPole DQN - https://www.youtube.com/watch?v=NP8pXZdU-5U&t=485s

DRL Snake - https://www.youtube.com/watch?v=PJl4iabBEz0&t=551s
