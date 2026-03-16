# AI Medical Imaging . Assignment 1 . Flipped Classroom Logs
This README contains the questions and short answers discussed during the flipped classroom sessions
## First Flipped Classroom
**Role:** Client
### Questions and Answers

#### Q1. Why is it necessary to already apply causal self-attention during training?  
**Answer:** It allows  to understand a sentence as a sentence instead of a combination of words. To understand the parts of sentence and how the correlation is between the words and the context of the word in a sentence. It is finetuning after the pretraining. 

#### Q2. GPT-models are trained to maximize the chance on text, not to make factually correct statements.? 
**Answer:** It is not based on actual facts but on probability. GPT models do not also know what a medical article is or not. The medical world is not stagnant, so new information is flowing in. 

#### Q3. During generating text with a GPT model, the "temperature" can be modified. Explain what happens with the output if you change the temperature higher or lower. 
**Answer:** It affects the probability; how sharp the probability is determined. It gives words with a lower probability the chance to be chosen. The temperature should be max 2. It is a design choice.  

#### Q4. What is the difference between top-k and top-p sampling? And how do you decide the right values for it? 
**Answer:** The same as for the temperature. If you train and validate, you can look at the outcomes and look at which gives the best outcome or which one you prefer. Top k is about fixed probability, and top P is more about flexibility. With top P you might sacrifice some accuracy. 

#### Q5. Why is it important to choose the right vocab size(number of unique tokens) and token size?
**Answer:** It depends on how efficient or how coherent you want it to be.  

#### Q6. At the end of a transformer, each possible subsequent token is assigned as a probability. Why isn't the token with the highest probability chosen by default?  
**Answer:** Ask the teacher.   Nothing really prevents this but with parameters like temperature, it will choose not always the highest probability to stimulate creativity. This is mostly useful in the context of a chatbot but might be less favorable in the medical context. 

#### Q7. What is the difference between, query, key and value? 
**Answer:** Value is the matrix parameter that takes into consideration both your query and key. Query is the input; the key is the different options. Value is the matching and the information you want to pass on, the actual samantic information. Query is the information I need from other patches. Key is the information I offer to other patches. Value is the actual information.

#### Q8. What are the risks and limitations of this model?
**Answer:** Limitations are design choices, the user can cause biases; the risks are for example whether the information is true or not.  You can have morally incorrect conversations with an AI bot, which can lead to dangerous choices.  

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
