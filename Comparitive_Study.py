#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import os
from stable_baselines3 import PPO

from time import sleep
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# In[2]:


environment_name = "ALE/Alien-v5"
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])


# # PPO Model

# In[3]:


#POLICY 1: MlpPolicy
model1 = PPO('MlpPolicy', env, verbose = 1)
model1.learn(total_timesteps=20000)


# In[4]:


from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model1, env, n_eval_episodes=10, render=True)


# In[5]:


#POLICY 2: CnnPolicy
model2 = PPO('CnnPolicy', env, verbose = 1)
model2.learn(total_timesteps=20000)


# In[6]:


#from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model2, env, n_eval_episodes=10, render=True)


# In[18]:


env.close()


# # A2C Model

# In[10]:


from stable_baselines3 import A2C


# In[8]:


env2 = gym.make(environment_name)


# In[14]:


model_a1 = A2C("MlpPolicy", env2, verbose=1)
model_a1.learn(total_timesteps=20000)


# In[22]:


evaluate_policy(model_a1, env2, n_eval_episodes=10, render=True)


# In[16]:


model_a2 = A2C("CnnPolicy", env2, verbose=1)
model_a2.learn(total_timesteps=20000)


# In[16]:


evaluate_policy(model_a2, env2, n_eval_episodes=10, render=True)


# # Saving Model

# In[20]:


path = os.path.join('Training', 'Saved Models', 'A2C_Model2')
model_a2.save(path)


# # Loading Model

# In[19]:


path = os.path.join('Training', 'Saved Models', 'A2C_Model1')
model_a1 = A2C.load(path, env=env2)


# # Testing

# In[26]:


episodes = 5
for episode in range(1, episodes+1):
    obs = env2.reset()
    done = False
    score = 0
    
    while not done:
        env2.render()
        action, _ = model_a2.predict(obs)
        obs, reward, done, info = env2.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))


# In[ ]:


env2.close()


# In[ ]:




