import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
from generate_data import client
import json


def llm_classify(prompt: str) -> int:
    response = client.chat.completions.create(
        model='gpt-4-0125-preview',
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "Please classify the prompt as appropriate or inappropriate. 0 for appropriate, 1 for inappropriate. Respond in json with key 'label' and an integer value of 0 or 1."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    response_data = json.loads(response.choices[0].message.content)

    return response_data['label']


def train_random_forest(X_train, y_train)-> None:

     # Create a bag of words representation of the data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    # Train a random forest classifier with default params
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    print("Trained Random Forest")

    filename = 'random_forest_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))


def train_logistic_regression(X_train, y_train) -> None:
    from sklearn.linear_model import LogisticRegression

    # Create a bag of words representation of the data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(random_state=42,class_weight='balanced')
    clf.fit(X_train, y_train)

    filename = 'logistic_regression_model.sav'
    pickle.dump(clf, open(filename, 'wb'))



def train_and_eval() -> None:
    df  = pd.read_csv('dataset.csv')

    positive_samples = df[df['label'] == 1] # inappropriate
    negative_samples = df[df['label'] == 0] # appropriate

    # Split the positive samples into train and test sets
    
    # We want at least 8 samples for training
    #TODO Remove this hardcoding of 8 samples
    X_positive_train = positive_samples['text'][:8]
    y_positive_train = positive_samples['label'][:8]
    X_positive_test = positive_samples['text'][8:]
    y_positive_test = positive_samples['label'][8:]

    # Split the negative samples into train and test sets
    X_negative_train, X_negative_test, y_negative_train, y_negative_test = train_test_split(
        negative_samples['text'], negative_samples['label'], test_size=0.2, random_state=42)

    # Combine the features and labels into two separate DataFrames in order to shuffle together
    train_positive = pd.DataFrame({'text': X_positive_train, 'label': y_positive_train})
    train_negative = pd.DataFrame({'text': X_negative_train, 'label': y_negative_train})

    test_positive = pd.DataFrame({'text': X_positive_test, 'label': y_positive_test})
    test_negative = pd.DataFrame({'text': X_negative_test, 'label': y_negative_test})

    # Concatenate the positive and negative samples and shuffle
    train_data = pd.concat([train_positive, train_negative]).sample(frac=1, random_state=42)
    test_data = pd.concat([test_positive, test_negative]).sample(frac=1, random_state=42)

    # Split the shuffled data back into features and labels
    X_train = train_data['text']
    y_train = train_data['label']

    X_test = test_data['text']
    y_test = test_data['label']
    print(y_test.shape)
    
    train_random_forest(X_train, y_train)
    train_logistic_regression(X_train, y_train)

    evaluate_models(X_test, y_test)


def evaluate_models(X_test, y_test) -> None:

    y_test = y_test.to_numpy()
    # Load and evaluate the random forest model
    filename = 'random_forest_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    X_test_vectors = vectorizer.transform(X_test)

    # Evaluate the random forest model
    accuracy_random_forest = clf.score(X_test_vectors, y_test)
    print(f"Accuracy Random Forest: {accuracy_random_forest}")
    # Add confusion matrix base don clf predictions
    y_pred_random_forest = clf.predict(X_test_vectors)

    cm = confusion_matrix(y_test, y_pred_random_forest, labels=[0, 1])
    print(cm, "confusion matrix")

    # Calculate precision and recall
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn)

    print(f"Precision Random Forest: {precision}")
    print(f"Recall Random Forest: {recall}")


    # Evaluate the logistic regression based approach
    filename = 'logistic_regression_model.sav'
    clf = pickle.load(open(filename, 'rb'))


    y_pred_logistic = clf.predict(X_test_vectors) # We can use the already vectorized X_test from the random forest model

    accuracy = sum([1 for i in range(len(y_test)) if y_test[i] == y_pred_logistic[i]]) / len(y_test)

    print(f"Accuracy Logistic Regression: {accuracy}")


    cm = confusion_matrix(y_test, y_pred_logistic , labels=[0, 1])

    # Calculate precision and recall
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    print(f"Precision Logistic Regression: {precision}")
    print(f"Recall Logistic Regression: {recall}")

    # Evaluate the LLM based approach using GPT 4

    y_pred_llm = [llm_classify(prompt) for prompt in X_test]

    accuracy_llm = sum([1 for i in range(len(y_test)) if y_test[i] == y_pred_llm[i]]) / len(y_test)

    print(f"Accuracy LLM: {accuracy_llm}")

    cm = confusion_matrix(y_test, y_pred_llm , labels=[0, 1])

    # Calculate precision and recall
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"Precision LLM: {precision}")
    print(f"Recall LLM: {recall}")

    


if __name__ == '__main__':
    train_and_eval()




