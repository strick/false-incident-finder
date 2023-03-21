import logging

import azure.functions as func
import pandas as pd
import numpy as np
import os as os
import re as re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# then compute similaties using cosine_sim with all other types to get a similartiy
def search(incident):

    # Choose min and max word sequesnces
    vectorizer = TfidfVectorizer(ngram_range=(1,5))

    #change this to only vectorize on known incidents
    false_incidents = incidents[incidents['FalseIncident'] == "true"]

    tfid_known_false_incidents = vectorizer.fit_transform(false_incidents['short_description'])

    if incident.name in false_incidents.index:
       return
    
    desc = clean_description(incident["short_description"])
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
    print("Importing data...")
    incidents = pd.read_csv(os.getcwd() + "/HttpTrigger1/incident_and_comments.csv")
    return incidents

def preprocess_data(incidents):

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
def add_false_incidents(df, feature, value):
    
    df.loc[(df[feature] == value), 'FalseIncident'] = "true"

from IPython.display import display, HTML

def print_comments(df):
    for i,row in incidents[incidents['quickly_closed'] == "true"].iterrows():
        display( HTML( pd.DataFrame({'comments_and_work_notes': [row['comments_and_work_notes']]}).to_html().replace("\\n","<br>") ) )

def get_similarity(incidents):
    
    # add a new column to the incidents dataframe that contains non-empty dataframes with similar incidents
    incidents["similar_incidents"] = incidents.apply(search, axis=1)

    non_empty_similar_incidents = incidents.dropna(subset=["similar_incidents"])
    similarity_scores = non_empty_similar_incidents["similar_incidents"].apply(lambda x: x["similarity_score"])

    return similarity_scores

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    import_data()
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

incidents = import_data()
incidents = preprocess_data(incidents)

incidents["quickly_closed"] = incidents.apply(is_quickly_closed, axis=1)

add_false_incidents(incidents, 'quickly_closed', 'true')
add_false_incidents(incidents, 'category', 'Hardware')

print(get_similarity(incidents))

