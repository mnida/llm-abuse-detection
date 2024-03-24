from typing import Literal
from train import llm_classify
import pickle
MethodType = Literal["random_forest", "llm", "logistic_regression"]

random_forest_model = pickle.load(open('random_forest_model.sav', 'rb'))
logistic_regression_model = pickle.load(open('logistic_regression_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))


'''
This function detects inappropriate prompts based on the classifier method. 
It outputs True if the prompt is inappropriate and False if the prompt is appropriate.

This function should be called between the API gateway and the LLM model input to filter out inappropriate prompts.
'''
def detect_abuse(prompt:str, method:MethodType) -> bool:
    if method == "random_forest":
        vectorized_prompt = vectorizer.transform([prompt])
        return bool(random_forest_model.predict(vectorized_prompt[0]))
    elif method == "llm":
        return bool(llm_classify(prompt))
    elif method == "logistic_regression":
        vectorized_prompt = vectorizer.transform([prompt])
        return bool(logistic_regression_model.predict(vectorized_prompt[0]))
    return True
    


