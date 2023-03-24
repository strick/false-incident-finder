import logging

import azure.functions as func
import pandas as pd
import numpy as np
import os as os
import re as re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

similarity_score_col = "short_description"
incidents = 0

# then compute similaties using cosine_sim with all other types to get a similartiy
def search(incident): #

    global incidents
    # Choose min and max word sequesnces
    vectorizer = TfidfVectorizer(ngram_range=(1,5))

    #change this to only vectorize on known incidents
    false_incidents = incidents[incidents['FalseIncident'] == "true"]

    tfid_known_false_incidents = vectorizer.fit_transform(false_incidents[similarity_score_col])

    if incident.name in false_incidents.index:
       return
    
    desc = clean_description(incident[similarity_score_col])
    query_vec = vectorizer.transform([desc]) 
    
    # compare the description to the knownIncidents list
    similarity = cosine_similarity(query_vec, tfid_known_false_incidents).flatten()

    # If there are anay items with a > .7 similarity, add this to the list
    indices = np.where(similarity > 0.7)[0]

    # Remove the current item from the list so that you odn't get a 1.0 similairty (i.e. itself)
    current_index = incident.name - 1
    indices = indices[indices != current_index]
    
    same_incident_indices = np.where(indices >= len(false_incidents))[0]
    indices = indices[indices < len(false_incidents)]

    results = false_incidents.iloc[indices].iloc[::-1]


    if not results.empty:
        # Add similarity score feature
        results["similarity_score"] = similarity[indices]

        return results

def import_data():
    global incidents
    print("Importing data...")
    incidents = pd.read_csv(os.getcwd() + "/HttpTrigger1/incident_and_comments.csv")
    return incidents

def preprocess_data():
    global incidents

    print("Preprocessing...")
    nanColumns = ['description', 'short_description']
    assignment_group = 'Custom Application Development'

    # Filter down to development team items
    incidents = incidents[incidents.assignment_group == assignment_group]

    # Remove unecssrary feature
    incidents = incidents.drop(['assignment_group', 'assigned_to', 'number', 'opened_by'], axis=1)

    # Remove all NaN values from dataset
    incidents = incidents.dropna(subset=nanColumns)

    # Remove bogus data
    incidents = incidents[incidents.description != 'asdf']

    # Remove incidents with non string date
    incidents = incidents[incidents['sys_created_on'].apply(lambda x: isinstance(x, str))]
    incidents = incidents[incidents['closed_at'].apply(lambda x: isinstance(x, str))]

    # Clean the short description and descriptions
    incidents.short_description = incidents.short_description.fillna(0)
    incidents.description = incidents.description.fillna(0)
    incidents["short_description"] = incidents["short_description"].apply(clean_description)
    incidents["description"] = incidents["description"].apply(clean_description)

    return incidents

# Function to clean special characters out of data
def clean_description(description):
    try:
        re.sub("[^a-zA-Z0-9 ]", "", description)
        return description
    except:
        #print(description)
        return description
    
# Get incidents that were closed in less than 60 minutes
from datetime import datetime
def is_quickly_closed(row):
    
    date1 = datetime.strptime(row["sys_created_on"], '%m/%d/%Y %I:%M:%S %p')
    date2 = datetime.strptime(row["closed_at"], '%m/%d/%Y %I:%M:%S %p')

    diff_minutes = int((date2 - date1).total_seconds() / 60)


    if diff_minutes <= 60:
        return "true"
                
    return "false"
    
# Helper function to set the FalseIncident column to true for all rows in the dataframe based on a feature and value.
def add_false_incidents(feature, value):
    global incidents
    incidents.loc[(incidents[feature] == value), 'FalseIncident'] = "true"

def get_similarity(incidentss):
    global incidents
    # add a new column to the incidents dataframe that contains non-empty dataframes with similar incidents
    incidents["similar_incidents"] = incidents.apply(search, axis=1)

    return _calculate_similarity(incidents)

def get_single_similarity(incidentss, user_submitted_incident):
     
    global incidents
    incidents["similar_incidents"] = user_submitted_incident.apply(search, axis=1)

    return _calculate_similarity(incidents)

def _calculate_similarity(incidentss):
    
    global incidents
    non_empty_similar_incidents = incidents.dropna(subset=["similar_incidents"])
    similarity_scores = non_empty_similar_incidents["similar_incidents"].apply(lambda x: x["similarity_score"])

    return similarity_scores

def runBN(train_data, user_submitted_incident):

    global incidents

    print("Running Naive Bayes on: " + user_submitted_incident)
    import nltk
    # Azure location
    nltk.data.path.append(os.getcwd() + "/nltk_data")
    from nltk.stem import WordNetLemmatizer
    stopwords = nltk.corpus.stopwords.words('english')

    # had to install this followed this guide:  https://stackoverflow.com/questions/13965823/resource-corpora-wordnet-not-found-on-heroku
    lemmatizer = WordNetLemmatizer()
    #nltk.download('stopwords')
    #train_data

    ## CREDIT:  https://www.analyticsvidhya.com/blog/2021/09/creating-a-movie-reviews-classifier-using-tf-idf-in-python/

    train_X_non = train_data['short_description']# + " " + train_data['description']   # '0' refers to the review text
    train_y = train_data['FalseIncident']   # '1' corresponds to Label (1 - positive and 0 - negative)
    test_X_non = incidents['short_description']# + " " + incidents['description']
    test_y = incidents['FalseIncident']
    train_X=[]
    test_X=[]

    #text pre processing
    for i in range(0, len(train_X_non)):
        review = re.sub('[^a-zA-Z]', ' ', train_X_non.iloc[i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        train_X.append(review)
        
    #text pre processing
    for i in range(0, len(test_X_non)):
        review = re.sub('[^a-zA-Z]', ' ', test_X_non.iloc[i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        test_X.append(review)

    #print(train_X[3])

    #tf idf
    tf_idf = TfidfVectorizer()
    #applying tf idf to training data
    X_train_tf = tf_idf.fit_transform(train_X)
    #applying tf idf to training data
    X_train_tf = tf_idf.transform(train_X)

    #print("n_samples: %d, n_features: %d" % X_train_tf.shape)

    #transforming test data into tf-idf matrix
    X_test_tf = tf_idf.transform(test_X)
    #print("n_samples: %d, n_features: %d" % X_test_tf.shape)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics

    # REF:  https://builtin.com/data-science/precision-and-recall
    #naive bayes classifier
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_tf, train_y)
    
    # Lets do a prediction
    test = [user_submitted_incident]# or ["avengers are here to stay in the world of us"]


    review = re.sub('[^a-zA-Z]', ' ', test[0])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    test_processed =[ ' '.join(review)]

    #test_processed = processText("This is unlike any kind of adventure movie my eyes")
    print("Processed incident value: " + test_processed[0])
    test_input = tf_idf.transform(test_processed)

    res=naive_bayes_classifier.predict(test_input)[0]
    print("Prediction: " + res)

    return res

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    global incidents

    #import_data()
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        
        result = testRun(name)
        
        #return func.HttpResponse(f"{single_similarity,bnResult}")
        return func.HttpResponse(result)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

def testRun(name):
    global incidents
    import_data()
    preprocess_data()

    incidents["quickly_closed"] = incidents.apply(is_quickly_closed, axis=1)

    add_false_incidents('quickly_closed', 'true')
    add_false_incidents('category', 'Hardware')

    user_submitted_incident = pd.DataFrame({
        "name": [-1],
        similarity_score_col: [name],#["Unable to reset NID"],
        "category": ["hardware"]
    })

    single_similarity = get_single_similarity(incidents, user_submitted_incident)
    
    if single_similarity.empty:
        single_similarity = "0"
    else:
        single_similarity = single_similarity.iloc[0].iloc[0].astype('str')

    print("Similarity score: " + single_similarity)
    res = runBN(incidents[incidents['FalseIncident'] == "true"], name)

   # print("Prediction: " + res)

    return "{'single_similarity': " + single_similarity + ",'prediction': " + res + "}"


obj = testRun("Unable to reset NID")
print(obj)
