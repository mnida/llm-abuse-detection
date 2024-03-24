# LLM Abuse and Appropriateness Detection

The goal of this project is to build a system that can classify whether a prompt is inappropriate or not. This is a common field in trust and safety for large foundation model providers and is an interesting problem to tackle because you are always adapting to bad actors and learning the model's potentially harmful outputs. This repo should demonstrate a few ideas around how to tackle this.

The main code is in generate_data.py and train.py and I wrote an example function in detection.py of what I would use if I were to spin up an API.

### Disclaimer: I am only using OpenAI models because I have many credits leftover from my startup. If given credits at Anthropic I would gladly use Claude for these tasks.

# Dataset Generation

As with all statistical learning methods, the data that we train on is our limiting factor and influences our model the most.

In this step I decided to generate my own data using an LLM and by handwriting some inappropriate prompts. This step has the highest risk of bias and is possibly the most import for determining what even consitutes an inappropriate prompt. However for the purpose of this project, I simply wrote a few terrible prompts that I thought an LLM should either refuse to answer or that should trigger an internal flag for the user. This was uncomfortable but necessary.

I could also have done data augmentation but I decided not to do this because there stil wouldn't be a variety of different words and phrases from doing this.

If I were to work on this project in an extended capacity, I would put the majority of my effort to make sure prompts were labeled correctly and that there was lots and lots of diverse data.

However, since this is a sample project I will continue with data on the order of 100 samples because I don't want to have to handwrite more inappropriate prompts.

# Three methods of classification

## 1. LLM classifies whether it is appropriate or not.

For an LLM, we will use an off the shelf model that is quick like GPT-3.

Pros: LLM's have a deep understanding of language and can be used to classify text data based on specific heuristics and diections that we can provide in the prompt. For instance we can ask it to classify anything related to violence and racism as inappropriate.

Cons:
LLMs are computationally expensive and slower than simpler models. Therefore this will greatly increase the throughput of our system if the prompt has to go through two LLMs before outputing the result.

In addition, we might simply put the instructions in the system prompt of the original LLM (Claude) to not give a response for certain queries and prompts. Therefore this could be seen as a waste of a step if the performance doesn't significantly improve.

## 2. Tree-Based Classifier Random Forest

Pros:
I chose a random forest (RF) because we have a very small dataset and RFs tend to have lower variance due to the random bagging of trees. If we had more data I would have choosen a gradient boosting model that could lower bias more.

Cons:
As I mentioned these models will have a higher bias which means lower accuracy, which might not be the best for a system that values precision.

In practice this performed horribly and only classified 0s. I would chalk this up to not enough data.

## 3. Logistic Regression Classifier

Pros: Computationally the cheapest, can be trained quickly and can work well with small datasets. In addition, it was easy for me to tweak the threshold of what is considered inappropriate, since the data imbalance pushes predictions towards 0 by default.

Cons: Not very complex and also potentially can have too high bias.

Overall we have a large class inbalance which will be present in most datasets of this nature.

## Note: I decided not to train a NN classifier (Logistic Regression with more layers pretty much) due to the limited amount of data, but this could be a valid strategy at a large scale.

## Results

Test set of 23 samples (10 inappropriate, 13 appropriate). While this is realistically too small to compare models, it did give insight to how well the LLM approach works.

| Approach            | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| LLM                 | 100%     | 100%      | 100%   |
| Random Forest       | 78.3%    | 0%        | 0%     |
| Logistic Regression | 82.6%    | 100%      | 20%    |

The random forest only predicted 0.

For this case I think precision is slightly more important than recall. Obviously recall is important because we don't want many cases where a FN is predicted, however everytime we have a FP we are angering users and potentially generating bad press.

Therefore precision and recall are a tradeoff between the average user experience and making sure no inappropriate prompts get through. I would slightly lean towards precision, under the assumption that our system is much better at catching really potentially harmful prompts like how to make a bomb but it has a higher chance of letting a prompt like "I hate you" through.

## Conclusion and Next Steps

Since the LLM approach performs the best, it is obvious that the LLMs (Claude and OpenAI) have already been trained/system prompted in order to be able to classify inappropriate prompts. However, there is most likely a type of prompt that the LLMs are not able to classify as inappropriate or abuse because they haven't seen them before. In this case the best solution would be to use the logistic regression model temporarily and then alert the pre-training and post-training researchers of the new set of illegal prompts so that they might bake it into the next base model and then we could remove the logistic regression model.

Further work could be done to improve the project including but not limited to:

- More data
- Tuning hyperparameters of models
- Building a monitoring tool for the detection function so that we can see what the model is classifying as inappropriate and use this for future training in the base LLM.
- Building a more complex model that can take into account the context of the prompt and the user. For instance, if the user has a history of inappropriate prompts, we might want to flag them more often.
