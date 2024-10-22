# Neural-Architecture-Search-on-Deep-Reinforcement-Learning
This is an attempt of Neural Architecture Search in Deep Reinforcement Learning and for a start I tried it with Lunar Lander and DQN network approximation. 
1. Reinforce rule with baseline for purpose of loss function from this paper on NAS [Paper link](https://arxiv.org/pdf/1611.01578)
2. Implemented ENAS by Google Brain for this. [Paper link](https://arxiv.org/abs/1802.03268)
3. Also this [DQN Tutorial](https://www.katnoria.com/nb_dqn_lunar/) was extremely helpful!
4. Have modified DQN with manual weights initialisation which is key in ENAS.
### Environment - Open AI GYM's Lunar-Lander-v2 
    Enviroment is solved if you reach 200 score for 100 episodes!

### ENAS Search Space : Search for best two layered feed forward nn possible from search space - 
Two layered feedforward neural networks with no skip connections  possible from dense layers of sizes (64,128,256,1024,2048) and activation functions (sigmoid,relu) => 100 architectures
### Reward -
(Didn't do much tuning into this, kind of arbitirary except that I have used average score/30 for normalisation purpose)
1. If the model converged then that trained model is run on different env seed for 500 episodes and average score is calculated, reward = average_score/30
2. If  model doesn't converge, reward = 1e-5
### Convergence - 
In our example after ~15 iterations it starts sampling  set of  only good performing architectures since our controller sampling policy is improved.Their performance will be comparable. I don't get a single best model obviously because our policy is stochastic and its higly unlikely that it will sample one best model after the end of training.
Plot - 
![plot](https://github.com/akjayant/Neural-Architecture-Search-Project/raw/master/controller_performance.png)

### Time Taken - 
~12 hrs on single NVIDIA GTx 1050 Ti  => 83.33% faster than brute force
(exhaustive search would have taken ~ 22 hrs)
Advantage is not quite big here because this is a simple example with search space including just 100 architectures, however this difference would have been huge if search space was big itself.
### Run to search
    python enas_contoller.py
    python plot_controller_performance.py
    
### Directly see one of the best model from our search space performing i.e, controller samples one of this at end of training

    sudo apt-get install ffmpeg
    cd best_model
    python play_lunar_video.py 
    
## Motivation - 
### 1) Impact of architecture complexity and choice of activation functions on DQN learning
*have been tested across various enviornment seeds
#### Low complexity and choice of activation function effects stability of training & convergence - 
1) 32,relu,64,relu
![p](https://github.com/akjayant/Neural-Architecture-Search-Project/raw/master/impact_of_architecture_choice_ddqn_results/model_env_seed_3/solved_200_32_64_3.png)
2) 1024,sigmoid,1024,sigmoid (didn't converge to 200 score)
![q](https://github.com/akjayant/Neural-Architecture-Search-Project/raw/master/impact_of_architecture_choice_ddqn_results/model_env_seed_4_sigmoid/solved_200_1024_1024_4.png)

#### Good architecture example - 
1) (1024,relu,1024,relu)
![w](https://github.com/akjayant/Neural-Architecture-Search-Project/raw/master/impact_of_architecture_choice_ddqn_results/model_env_seed_4/solved_200_1024_1024_4.png)
#### Overly complex architectures are also not good and sometimes take too much time to train
2) (2048,relu,2048,relu)
![rr](https://github.com/akjayant/Neural-Architecture-Search-Project/raw/master/impact_of_architecture_choice_ddqn_results/model_env_seed_4/solved_200_2048_2048_4.png)
#### 2) NAS techniques till now are not good enough for finding a global optimal or state of the art architecture and it also have human bias but it does help in finding the best from a given search space efficiently and since neural networks used in RL applications till now are not overly complex this can be a good option! 
## Challenge remains to be addressed! 
1. Randomness in training data due to epsilon greedy approach to account for exploitaion-exploration trade off.
2. Scalability issue of this for more complex & large networks like CNNs still exist where training a single CNN takes lot of time.
