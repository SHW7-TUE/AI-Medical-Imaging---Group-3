# AI Medical Imaging . Assignment 1 . Flipped Classroom Logs
This README contains the questions and short answers discussed during the flipped classroom sessions
## First Flipped Classroom
**Role:** Client
### Questions and Answers

#### Q1. 
**Answer:** 

#### Q2. 
**Answer:** 

#### Q3. 
**Answer:** 





---
**Role:** Consultant
### Questions and Answers

#### Q1. What happens if you remove the feed-forward part?
**Answer:** Without the feed-forward network, only self-attention remains. The model becomes less powerful, can learn fewer complex patterns, and the generated text will be less coherent.

#### Q2. Can you include explanations of medical processes in the data so the model can generate more explanations?
**Answer:** Yes, the model will learn to generate that type of explanation. However, it remains a language model without real medical understanding, so the explanations may sound plausible but still be incorrect.

#### Q3. How do you know if your dataset is large enough?
**Answer:** Look at the difference between training and validation loss. A large gap indicates overfitting and possibly too little data. Poor generalization and repetition in the generated output are also warning signs.

#### Q4. What should you do if the model generates nonsense?
**Answer:** Adjust sampling settings such as temperature and top_k, and check the training and hyperparameters. Change one parameter at a time and evaluate its effect.
