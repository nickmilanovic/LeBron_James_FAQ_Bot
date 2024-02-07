"""
Nicholas Milanovic
February 6, 2023
Sam Scott, Mohawk College, May 2021
(Modified March 16, 2022: utilizing given resources to make a FAQ bot on
LeBron James)
"""

#import regex as re
#import spacy
#from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from joblib import load
clf = load('classifier.joblib')
vect = load('vectorizer.joblib')
import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
import openai
openai.api_key = "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"
record = ""


def file_input(filename):
    """Loads each line of the file into a list and returns it."""
    lines = []
    with open(filename) as file: # opens the file and assigns it to a variable
        for line in file:
            lines.append(line.strip()) # the strip method removes extra whitespace
    return lines

def understand(utterance):
    """This method processes an utterance to determine which intent it
    matches. The index of the intent is returned, or -1 if no intent
    is found, or -2 if it is determined that there are no similarities to the
    intents and if a question is not being asked."""

    global intents # declare that we will use a global variable

    # setting up a TfidVectorizer with ngram_range of (1,2) and using lebron
    # as a stop word
    stop_words = ["lebron"]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words=stop_words)
    vectors = vectorizer.fit_transform(intents)
    new_vector = vectorizer.transform([utterance])

    # calculation of cosine similarity and euclidean distances
    cosine_sim = cosine_similarity(new_vector, vectors)
    euclidean_sim = euclidean_distances(new_vector, vectors)
    #print("Cosine Similarities: ", cosine_sim)
    #print("Euclidean Distance: ", euclidean_sim)

    # NLP used to determine if a question is being asked or not
    new_doc = nlp(utterance)
    match = Matcher(nlp.vocab)
    pattern = [
            {"LOWER":{"IN":["do","who","what","when","where","how","does","why","is","are","?"]}},
            {"POS": "PUNCT","OP":"*"}
    ]
    match.add("question", [pattern])
    question_match = match(new_doc)

    if all(val >= 0.99 for val in euclidean_sim[0]) and not question_match:
        return -2
    # if a question is being asked, check for similarities in intents using
    # euclidean distance and cosine similarities
    for token in new_doc:
        if question_match:
            max_cos = cosine_sim[0]
            max_cos = np.max(cosine_sim)
            min_euc = euclidean_sim[0]
            min_euc = np.min(euclidean_sim)
            maximum_value = np.where(cosine_sim == max_cos)
            minimum_value = np.where(euclidean_sim == min_euc)
            if int(maximum_value[1][0]) != 0 and max_cos >= 0.5 and min_euc <= 1:
                return int((maximum_value[1][0])/2)
            elif max_cos < 0.5 and min_euc >= 1:
                return -1
            elif int(maximum_value[1][0]) == 0:
                return 0


def generate(intent, utterance):
    """This function returns an appropriate response given a user's
    intent."""

    global responses # declare that we will use a global variable

    if intent >= 0 and utterance != "goodbye".casefold() and utterance != "hello".casefold():
        return responses[intent]
    elif intent == -1 and utterance != "goodbye".casefold() and utterance != "hello".casefold():
        doc = nlp(utterance)
        matcher = Matcher(nlp.vocab)
        question_pattern = [
            {"LOWER":{"IN":["who","what","when","where","how","does","why","is","are","?"]}},
            {"POS": "PUNCT","OP":"*"}
        ]
        matcher.add("question", [question_pattern])
        question_matches = matcher(doc)
        for token in doc:
            if question_matches:
                openai_response = openai.Completion.create(engine="text-davinci-002",
                prompt=utterance,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,)
                response_text = openai_response.choices[0].text.strip()
                res = "I'm not 100% sure of this, but " + response_text + " But lets stay on the topic of LeBron James."
                return res
        res = "Sorry, I don't know the answer to that"
        return res
    elif intent == -2 and utterance != "goodbye".casefold() and utterance != "hello".casefold():
        vect_utterance = vect.transform([utterance])
        prediction = clf.predict(vect_utterance)
        for pred in prediction:
            if prediction[0] == 1:
                res = "That's good to hear!"
                return res
            elif prediction[0] == 0:
                res = "Oh no, please contact our support email for someone to rectify the matter."
                return res

intents = file_input("questions.txt")
responses = file_input("answers.txt")


def chat():
    # talk to the user

    print()
    utterance = ""
    while True:
        utterance = input(">>> ")
        if utterance == 'goodbye':
            break
        if utterance == 'hello'.casefold():
            print("Hello! I know stuff about LeBron James. When you're done talking, just say 'goodbye'.")
        intent = understand(utterance)
        response = generate(intent, utterance)
        print(response)
        print()

    print("Nice talking to you!")