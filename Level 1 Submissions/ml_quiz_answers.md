# 1. What is supervised learning? Provide an example. 
- Supervised learning is a type of machine learning where the model is provided with inputs and corresponding outputs (labels) during training. The goal is to learn a mapping from inputs to outputs so that it can predict the correct output for new, unseen data.  

- Example: In a classification task, the input data could be images of animals, and the outputs (labels) would be categories like “cat” or “dog.” The model is trained on labeled examples like `(image1, "cat")`, `(image2, "dog")` and, after training, can predict the label of a new image it hasn’t seen before.  

---------------------------------------------------------------------------------------------------------------------------------
# 2. Explain the difference between supervised and unsupervised learning.  
- Supervised Learning:
   - In supervised learning, the model is trained on labeled data, where both inputs and outputs are provided.  
   - The goal is to learn a relationship between inputs and outputs for prediction or classification.  
   - Example: Predicting housing prices based on features like size, location, and number of rooms.  

- Unsupervised Learning:  
   - In unsupervised learning, the model is provided only with input data and no labels.  
   - The goal is to find patterns, structures, or groupings within the data.  
   - Example: Clustering customers based on their purchasing behavior to find similar groups.  

---------------------------------------------------------------------------------------------------------------------------------

# 3. How is reinforcement learning different from supervised learning?  
-Reinforcement learning (RL) differs from supervised learning in how the model learns:  

- In supervised learning, the model learns from labeled examples where the correct output is provided for each input. The goal is to minimize the difference between predictions and correct labels.  

- In reinforcement learning, there are no explicit labels. Instead, an agent interacts with an environment, takes actions, and learns through **rewards** and penalties. The goal is to maximize long-term rewards by learning an optimal policy.  

- Example of RL: Training a robot to navigate a maze. The robot learns through trial and error by receiving rewards for reaching the correct path and penalties for hitting walls.  
