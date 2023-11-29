# Predicting False Incident Requests
CAP 5610 - Brian Strickland (1368280)

- Collect the data
- Clean the data
- Feature extract to find known false incidnets
- Use a Fequency Matrix to create similarties between incidents base don short / long descrition   (do experiements with toehr columsn)
- 

## Collect Data to build training data
Data was created by generating a custom service portal page within ServiceNow and placing a Data Table by Instance widget on to the page with defined columns.  Over 200K incidents were then able to be exported.


```python
import pandas as pd
import numpy as np

#pd.options.display.float_format = '{:.20f}'.format


#incidents = pd.read_csv("incident_full.csv")
incidents = pd.read_csv("incident_and_comments.csv")

#incidents

```

### Preprocess data

#### Drop Data
- Remove unused colums from the data
- Only use rows that pertain to the Custom Application Development group
- Remove bogus data and hand NaN values


```python
import re

nanColumns = ['description', 'short_description']
assignment_group = 'Custom Application Development'

# Filter down to development team items
incidents = incidents[incidents.assignment_group == assignment_group]

# Remove unecssrary feature
#incidents = incidents.drop(['assignment_group', 'assigned_to', 'number', 'opened_by'], axis=1)
incidents = incidents.drop(['number', 'opened_by'], axis=1)

# Remove all NaN values from dataset
incidents = incidents.dropna(subset=nanColumns)

# Remove bogus data
incidents = incidents[incidents.description != 'asdf']

# Remove incidents with non string date
incidents = incidents[incidents['sys_created_on'].apply(lambda x: isinstance(x, str))]
incidents = incidents[incidents['closed_at'].apply(lambda x: isinstance(x, str))]
```

#### Clean Up Data


```python
# Function to clean special characters out of data
def clean_description(description):
    try:
        re.sub("[^a-zA-Z0-9 ]", "", description)
        return description
    except:
        #print(description)
        return description
        
# Clean the short description and descriptions
incidents.short_description = incidents.short_description.fillna(0)
incidents.description = incidents.description.fillna(0)
incidents["short_description"] = incidents["short_description"].apply(clean_description)
incidents["description"] = incidents["description"].apply(clean_description)
incidents["FalseIncident"] = "False"

# set random seed for reproducibility
#np.random.seed(143)

# generate a random number between 0 and the length of the dataframe
#num_true = np.random.randint(0, len(incidents))

# set that many incidents to True for the "FalseIncident" column
#incidents.loc[np.random.choice(incidents.index, size=num_true), "FalseIncident"] = "true"

```


```python
incidents
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>short_description</th>
      <th>description</th>
      <th>assignment_group</th>
      <th>u_inc_dept</th>
      <th>sys_created_on</th>
      <th>closed_at</th>
      <th>comments_and_work_notes</th>
      <th>FalseIncident</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Software</td>
      <td>The SAFE Form website is not properly generati...</td>
      <td>When individuals complete a SAFE Form, there i...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>09/25/2017 04:07:49 PM</td>
      <td>10/13/2017 08:48:07 AM</td>
      <td>10/13/2017 08:48:07 AM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Software</td>
      <td>Undergraduate Admissions Web App (OLA) file wa...</td>
      <td>The web application load in PS failed because ...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>10/25/2017 08:26:34 AM</td>
      <td>10/30/2017 02:48:11 PM</td>
      <td>10/30/2017 02:48:11 PM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hardware</td>
      <td>Unable to access http://directory.sdes.ucf.edu...</td>
      <td>Unable to access site</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>10/27/2017 08:55:53 AM</td>
      <td>11/01/2017 11:48:06 AM</td>
      <td>11/01/2017 11:48:06 AM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Software</td>
      <td>We are unable to enter redeemed vouchers into ...</td>
      <td>This is the first time this year that we are p...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>10/27/2017 02:45:07 PM</td>
      <td>11/01/2017 03:48:07 PM</td>
      <td>11/01/2017 03:48:07 PM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Software</td>
      <td>UA forms such as Residency, Reacts, Counselor ...</td>
      <td>UA forms such as Residency, Reacts, Counselor ...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>10/31/2017 08:39:06 AM</td>
      <td>11/03/2017 01:48:08 PM</td>
      <td>11/03/2017 01:48:08 PM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>351</th>
      <td>Software</td>
      <td>Trying to submit changes to UCF Phonebook and ...</td>
      <td>Details:  Althea Robinson called to report she...</td>
      <td>Custom Application Development</td>
      <td>CCIE ADMINISTRATION</td>
      <td>05/18/2022 11:55:32 AM</td>
      <td>05/23/2022 04:48:07 PM</td>
      <td>05/23/2022 04:48:07 PM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>352</th>
      <td>Software</td>
      <td>I need access to following link below. When I ...</td>
      <td>User's relationship to UCF: Employee\n\nUser's...</td>
      <td>Custom Application Development</td>
      <td>COLLEGE OF BUSINESS DEAN</td>
      <td>05/25/2022 08:57:12 AM</td>
      <td>05/31/2022 02:48:00 PM</td>
      <td>05/31/2022 02:48:00 PM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>353</th>
      <td>Software</td>
      <td>Cannot log into COBA Test Management</td>
      <td>I have tried several times to log in but keep ...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>06/03/2022 04:46:22 AM</td>
      <td>06/23/2022 10:48:00 AM</td>
      <td>06/23/2022 10:48:00 AM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>354</th>
      <td>Software</td>
      <td>Knights Email Acount Login and Password Reset/...</td>
      <td>Incoming student Sydney Schumacher called in f...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>06/14/2022 01:31:45 PM</td>
      <td>06/20/2022 08:48:05 AM</td>
      <td>06/20/2022 08:48:05 AM - System (Additional co...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>355</th>
      <td>Software</td>
      <td>custom app e911.it.ucf.edu not pulling data</td>
      <td>custom app is displaying datatables error when...</td>
      <td>Custom Application Development</td>      
      <td>UCF IT</td>
      <td>06/30/2022 03:48:37 PM</td>
      <td>07/18/2022 01:48:00 PM</td>
      <td>07/18/2022 01:48:00 PM - System (Additional co...</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>355 rows × 11 columns</p>
</div>



### Feature Extraction
Here we will look at various methods to identify known false incidents and add a feature, FalseIncident, and add this to each of those rows.   The below method will enable the ability to quickly mark an identify row(s) as a false incident.



```python
# Helper function to set the FalseIncident column to true for all rows in the dataframe based on a feature and value.
def add_false_incidents(df, feature, value):
    
    df.loc[(df[feature] == value), 'FalseIncident'] = "true"
```

### Quickly Closed Incidents

Here we will explore incidents that were closed quickly (within an hour) and if any are found, deteremine if they are truly false incidents.  From this we can gather some information about what makes quickly closed incidents false incidents that we'll add into our scoring for identifying false incidents.


```python
# Get incidents that were closed in less than 60 minutes
from datetime import datetime
def is_quickly_closed(row):
    
    date1 = datetime.strptime(row["sys_created_on"], '%m/%d/%Y %I:%M:%S %p')
    date2 = datetime.strptime(row["closed_at"], '%m/%d/%Y %I:%M:%S %p')

    diff_minutes = int((date2 - date1).total_seconds() / 60)


    if diff_minutes <= 60:
        return "true"
            
    return "false"
        

incidents["quickly_closed"] = incidents.apply(is_quickly_closed, axis=1)

```


```python
incidents[incidents['quickly_closed'] == "true"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>short_description</th>
      <th>description</th>
      <th>assignment_group</th>
      <th>u_inc_dept</th>
      <th>sys_created_on</th>
      <th>closed_at</th>
      <th>comments_and_work_notes</th>
      <th>FalseIncident</th>
      <th>quickly_closed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>326</th>
      <td>Software</td>
      <td>Unable to reset NID password</td>
      <td>User called in to report that he has been unab...</td>
      <td>Custom Application Development</td>
      <td>CCIE DEAN</td>
      <td>12/09/2021 09:33:23 AM</td>
      <td>12/09/2021 09:49:04 AM</td>
      <td>12/09/2021 09:49:04 AM - Yacine Tazi (Addition...</td>
      <td>False</td>
      <td>true</td>
    </tr>
  </tbody>
</table>
</div>



From here we can tell that there is only a very small number of incidents that get closed in less than an hour.   With this specific incident, it's a password reset, so lets look at the comments to see if it was user error or if the support center help them to reset the password.


```python
from IPython.display import display, HTML

def print_comments(df):
    for i,row in incidents[incidents['quickly_closed'] == "true"].iterrows():
        display( HTML( pd.DataFrame({'comments_and_work_notes': [row['comments_and_work_notes']]}).to_html().replace("\\n","<br>") ) )

```

Based on the following, we can deteremine that this wasn't an incident that needed to be resolved by the development team, but rather an expected outage which eventually enable the customer to login:

1) There is an intermittent outage with Self Service Reset Tool when users are trying to reset their password using email. This is not consistent behavior and will resolve itself shortly.
2) Was able to log in again

With this information, we can go ahead and flag this particular incident as a FalseIncident


```python
with pd.option_context('display.max_colwidth', None):
    print(incidents[incidents['quickly_closed'] == "true"].comments_and_work_notes)
```

    326    12/09/2021 09:49:04 AM - Yacine Tazi (Additional comments)\nWas able to log in again\n\n12/09/2021 09:47:40 AM - System (Additional comments)\nWe are continuing to investigate the underlying issue:\n\n\nWHAT IS HAPPENING?\nThere is an intermittent outage with Self Service Reset Tool when users are trying to reset their password using email. This is not consistent behavior and will resolve itself shortly. \n\nWHO IS IMPACTED?\nAnyone that need to reset their NID password using the email functionality. \n\nWHAT ARE WE DOING ABOUT IT?\nWe are currently investigating this issue.\n\nWHAT HAPPENS NEXT?\nWe are currently investigating and will keep everyone posted once the issue is resolved. \n\nWHAT DO I NEED TO DO?\nShould users encounter this issue during password reset, please wait 15-20 minutes and try again. \n\n\n\n12/09/2021 09:34:35 AM - Diego Cruces (Work notes)\nRouting to the Custom Application Development team for further investigation.\n\n
    Name: comments_and_work_notes, dtype: object


### Add False Incident Feature


```python
add_false_incidents(incidents, 'quickly_closed', 'true')
incidents[incidents['FalseIncident'] == "true"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>short_description</th>
      <th>description</th>
      <th>assignment_group</th>
      <th>u_inc_dept</th>
      <th>sys_created_on</th>
      <th>closed_at</th>
      <th>comments_and_work_notes</th>
      <th>FalseIncident</th>
      <th>quickly_closed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>326</th>
      <td>Software</td>
      <td>Unable to reset NID password</td>
      <td>User called in to report that he has been unab...</td>
      <td>Custom Application Development</td>
      <td>CCIE DEAN</td>
      <td>12/09/2021 09:33:23 AM</td>
      <td>12/09/2021 09:49:04 AM</td>
      <td>12/09/2021 09:49:04 AM - Yacine Tazi (Addition...</td>
      <td>true</td>
      <td>true</td>
    </tr>
  </tbody>
</table>
</div>



### Category type is hardware
Incidents that are set as hardware are assumed to be a false incident since the software development team doesn't deal with hardware issues.


```python
add_false_incidents(incidents, 'category', 'Hardware')
#incidents[incidents['FalseIncident'] == "true"]

# Set the selected indices to True
#incidents.loc[incidents['FalseIncident'] == False, 'FalseIncident'] = np.random.choice([True, False], size=incidents['FalseIncident'].shape[0], p=[0.5, 0.5])

# store to training data
train_data = incidents[incidents['FalseIncident'] == "true"].copy()
train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>short_description</th>
      <th>description</th>
      <th>assignment_group</th>
      <th>u_inc_dept</th>
      <th>sys_created_on</th>
      <th>closed_at</th>
      <th>comments_and_work_notes</th>
      <th>FalseIncident</th>
      <th>quickly_closed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Hardware</td>
      <td>Unable to access http://directory.sdes.ucf.edu...</td>
      <td>Unable to access site</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>10/27/2017 08:55:53 AM</td>
      <td>11/01/2017 11:48:06 AM</td>
      <td>11/01/2017 11:48:06 AM - System (Additional co...</td>
      <td>true</td>
      <td>false</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hardware</td>
      <td>Students are unable to upload forms to online ...</td>
      <td>Students are required to upload their involvem...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>12/04/2017 01:13:26 PM</td>
      <td>12/12/2017 11:48:05 AM</td>
      <td>12/12/2017 11:48:06 AM - System (Additional co...</td>
      <td>true</td>
      <td>false</td>
    </tr>
    <tr>
      <th>121</th>
      <td>Hardware</td>
      <td>The Exchange Unified Messaging voicemail assig...</td>
      <td>The Exchange Unified Messaging voicemail assig...</td>
      <td>Custom Application Development</td>
      <td>UCF IT</td>
      <td>03/13/2019 06:38:29 AM</td>
      <td>03/21/2019 03:48:13 PM</td>
      <td>03/21/2019 03:48:13 PM - System (Additional co...</td>
      <td>true</td>
      <td>false</td>
    </tr>
    <tr>
      <th>210</th>
      <td>Hardware</td>
      <td>Can't access DHCP reservations or do anythong ...</td>
      <td>I got the new URL my.it.ucf.edeu and I can get...</td>
      <td>Custom Application Development</td>
      <td>FINANCIAL AFFAIRS</td>
      <td>02/19/2020 09:41:48 AM</td>
      <td>02/24/2020 12:48:09 PM</td>
      <td>02/24/2020 12:48:09 PM - System (Additional co...</td>
      <td>true</td>
      <td>false</td>
    </tr>
    <tr>
      <th>292</th>
      <td>Hardware</td>
      <td>We are not able to access lead.sdes.ucf.edu/ad...</td>
      <td>All computers in the office are getting the sa...</td>
      <td>Custom Application Development</td>
      <td>SDES STU LEADERSHIP DEVELOP</td>
      <td>08/19/2021 08:47:17 AM</td>
      <td>08/26/2021 11:48:14 AM</td>
      <td>08/26/2021 11:48:14 AM - System (Additional co...</td>
      <td>true</td>
      <td>false</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Software</td>
      <td>Unable to reset NID password</td>
      <td>User called in to report that he has been unab...</td>
      <td>Custom Application Development</td>
      <td>CCIE DEAN</td>
      <td>12/09/2021 09:33:23 AM</td>
      <td>12/09/2021 09:49:04 AM</td>
      <td>12/09/2021 09:49:04 AM - Yacine Tazi (Addition...</td>
      <td>true</td>
      <td>true</td>
    </tr>
  </tbody>
</table>
</div>



## Using Cosine Similarity 
### Create a Feature Matrix
Here we'll create a feature matrix based on the short description values of known incidents (i.e. FalseIncident == "true").   From there we can create a similarite score on all other incidents that have been submitted to see if we can identify some other false incidents.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

def createTfid(train_data):
    # Choose min and max word sequesnces
    vectorizer = TfidfVectorizer(ngram_range=(1,5))

    #change this to only vectorize on known incidents
    #false_incidents = incidents[incidents['FalseIncident'] == "true"]
    false_incidents = train_data.copy()

    tfid_known_false_incidents = vectorizer.fit_transform(false_incidents['short_description'] )
    return tfid_known_false_incidents, false_incidents, vectorizer


# Choose min and max word sequesnces
#vectorizer = TfidfVectorizer(ngram_range=(1,5))

#change this to only vectorize on known incidents
#false_incidents = train_data.copy()

#tfid_known_false_incidents = vectorizer.fit_transform(false_incidents['short_description'] )
tfid_known_false_incidents, false_incidents, vectorizer = createTfid(train_data)

```


```python
tfid_known_false_incidents
```




    <6x316 sparse matrix of type '<class 'numpy.float64'>'
    	with 339 stored elements in Compressed Sparse Row format>



### Generate the Similarites


```python
from sklearn.metrics.pairwise import cosine_similarity
# then compute similaties using cosine_sim with all other types to get a similartiy
def search(incident, tfid_known_false_incidents, false_incidents, vectorizer):

    if incident.name in false_incidents.index:
       return
    
    desc = clean_description(incident["short_description"])
    query_vec = vectorizer.transform([desc]) 
    
    # compare the description to the knownIncidents list
    similarity = cosine_similarity(query_vec, tfid_known_false_incidents).flatten()

    # If there are anay items with a > .7 similarity, add this to the list
    indices = np.where(similarity > 0.5)[0]

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

# add a new column to the incidents dataframe that contains non-empty dataframes with similar incidents
#incidents["similar_incidents"] = incidents.apply(search, axis=1)
#incidents["similar_incidents"] = incidents.apply(search, args=(tfid_known_false_incidents), axis=1)
incidents["similar_incidents"] = incidents.apply(search, args=(tfid_known_false_incidents,false_incidents, vectorizer ,), axis=1)



non_empty_similar_incidents = incidents.dropna(subset=["similar_incidents"])
similarity_scores = non_empty_similar_incidents["similar_incidents"].apply(lambda x: x["similarity_score"])
```


```python
non_empty_similar_incidents
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>short_description</th>
      <th>description</th>
      <th>assignment_group</th>
      <th>u_inc_dept</th>
      <th>sys_created_on</th>
      <th>closed_at</th>
      <th>comments_and_work_notes</th>
      <th>FalseIncident</th>
      <th>quickly_closed</th>
      <th>similar_incidents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>212</th>
      <td>Software</td>
      <td>User called in to report that he is unable to ...</td>
      <td>User called in to report that he is unable to ...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>02/24/2020 12:45:58 PM</td>
      <td>04/10/2020 08:48:08 AM</td>
      <td>04/10/2020 08:48:08 AM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>268</th>
      <td>Software</td>
      <td>Unable to reset NID Password due to webpage ou...</td>
      <td>User's password has expired and they attempted...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>03/08/2021 04:48:03 PM</td>
      <td>03/15/2021 01:48:02 PM</td>
      <td>03/15/2021 01:48:02 PM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>272</th>
      <td>Software</td>
      <td>NID Password reset for account (NID): da909465</td>
      <td>User is having issues being able to reset his ...</td>
      <td>Custom Application Development</td>
      <td>HOSPITALITY MANAGEMENT DEAN</td>
      <td>03/09/2021 01:30:01 PM</td>
      <td>03/15/2021 01:48:08 PM</td>
      <td>03/15/2021 01:48:08 PM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Software</td>
      <td>User is unable to reset knights mail due to it...</td>
      <td>User states when trying to reset his password ...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>12/07/2021 07:31:51 PM</td>
      <td>12/14/2021 02:48:04 PM</td>
      <td>12/14/2021 02:48:04 PM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Software</td>
      <td>Unable to reset NID password</td>
      <td>User called in to report that she has been una...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>12/09/2021 09:27:13 AM</td>
      <td>01/21/2022 08:48:06 AM</td>
      <td>01/21/2022 08:48:06 AM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>335</th>
      <td>Software</td>
      <td>Students are not able to upload documents to t...</td>
      <td>Students are trying to go to their profile and...</td>
      <td>Custom Application Development</td>
      <td>SDES STU LEADERSHIP DEVELOP</td>
      <td>01/18/2022 03:53:46 PM</td>
      <td>02/09/2022 08:48:15 AM</td>
      <td>02/09/2022 08:48:15 AM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category                                 ...</td>
    </tr>
  </tbody>
</table>
</div>



### Similarity Scores
The table below displays each incident that has a similarity score of >=0.7 with any known incident.   Each additional NaN column is just a known_incident that the current row has no simiality with


```python
similarity_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>326</th>
      <th>292</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>212</th>
      <td>0.536588</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>268</th>
      <td>0.988003</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>272</th>
      <td>0.517992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>318</th>
      <td>0.506493</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>325</th>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>335</th>
      <td>NaN</td>
      <td>0.513134</td>
    </tr>
  </tbody>
</table>
</div>




```python
#non_empty_similar_incidents['similar_incidents']
```

## Finding False Incidents Through Recursive Similarity


```python
def findFalseOnes(df):
    
    new = incidents.copy()
    tfid_known_false_incidents, false_incidents, vectorizer = createTfid(df)
    
    new["similar_incidents"] = new.apply(search, args=(tfid_known_false_incidents,false_incidents, vectorizer ,), axis=1)
    not_empty = new.dropna(subset=["similar_incidents"])
    sim_scores = not_empty["similar_incidents"].apply(lambda x: x["similarity_score"])
    
    return not_empty, sim_scores

p = non_empty_similar_incidents.drop(['similar_incidents'], axis=1)
#e,b = findFalseOnes(p)

count = 0
df_found = pd.DataFrame()
df_sim = pd.DataFrame()
while(not p.empty):
    p, b = findFalseOnes(p)
    
    if(p.empty):
        break
    count = count + p.shape[0]
    df_found = df_found.append(p)
    df_sim = df_sim.append(b)

    p = df_found


print(count)
df_found
    
```

    /tmp/ipykernel_35266/1471197761.py:24: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_found = df_found.append(p)
    /tmp/ipykernel_35266/1471197761.py:25: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_sim = df_sim.append(b)
    /tmp/ipykernel_35266/1471197761.py:24: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_found = df_found.append(p)
    /tmp/ipykernel_35266/1471197761.py:25: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_sim = df_sim.append(b)


    7





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>short_description</th>
      <th>description</th>
      <th>assignment_group</th>
      <th>u_inc_dept</th>
      <th>sys_created_on</th>
      <th>closed_at</th>
      <th>comments_and_work_notes</th>
      <th>FalseIncident</th>
      <th>quickly_closed</th>
      <th>similar_incidents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>292</th>
      <td>Hardware</td>
      <td>We are not able to access lead.sdes.ucf.edu/ad...</td>
      <td>All computers in the office are getting the sa...</td>
      <td>Custom Application Development</td>
      <td>SDES STU LEADERSHIP DEVELOP</td>
      <td>08/19/2021 08:47:17 AM</td>
      <td>08/26/2021 11:48:14 AM</td>
      <td>08/26/2021 11:48:14 AM - System (Additional co...</td>
      <td>true</td>
      <td>false</td>
      <td>category                                 ...</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Software</td>
      <td>Unable to reset NID password</td>
      <td>User called in to report that he has been unab...</td>
      <td>Custom Application Development</td>
      <td>CCIE DEAN</td>
      <td>12/09/2021 09:33:23 AM</td>
      <td>12/09/2021 09:49:04 AM</td>
      <td>12/09/2021 09:49:04 AM - Yacine Tazi (Addition...</td>
      <td>true</td>
      <td>true</td>
      <td>category                                 ...</td>
    </tr>
    <tr>
      <th>212</th>
      <td>Software</td>
      <td>User called in to report that he is unable to ...</td>
      <td>User called in to report that he is unable to ...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>02/24/2020 12:45:58 PM</td>
      <td>04/10/2020 08:48:08 AM</td>
      <td>04/10/2020 08:48:08 AM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>268</th>
      <td>Software</td>
      <td>Unable to reset NID Password due to webpage ou...</td>
      <td>User's password has expired and they attempted...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>03/08/2021 04:48:03 PM</td>
      <td>03/15/2021 01:48:02 PM</td>
      <td>03/15/2021 01:48:02 PM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Software</td>
      <td>User is unable to reset knights mail due to it...</td>
      <td>User states when trying to reset his password ...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>12/07/2021 07:31:51 PM</td>
      <td>12/14/2021 02:48:04 PM</td>
      <td>12/14/2021 02:48:04 PM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Software</td>
      <td>Unable to reset NID password</td>
      <td>User called in to report that she has been una...</td>
      <td>Custom Application Development</td>
      <td>NaN</td>
      <td>12/09/2021 09:27:13 AM</td>
      <td>01/21/2022 08:48:06 AM</td>
      <td>01/21/2022 08:48:06 AM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category              short_description  ...</td>
    </tr>
    <tr>
      <th>335</th>
      <td>Software</td>
      <td>Students are not able to upload documents to t...</td>
      <td>Students are trying to go to their profile and...</td>
      <td>Custom Application Development</td>
      <td>SDES STU LEADERSHIP DEVELOP</td>
      <td>01/18/2022 03:53:46 PM</td>
      <td>02/09/2022 08:48:15 AM</td>
      <td>02/09/2022 08:48:15 AM - System (Additional co...</td>
      <td>False</td>
      <td>false</td>
      <td>category                                 ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sim
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>335</th>
      <th>325</th>
      <th>268</th>
      <th>326</th>
      <th>292</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>292</th>
      <td>0.553778</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>326</th>
      <td>NaN</td>
      <td>0.543881</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>212</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.649939</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>268</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.984638</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>318</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.595599</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>325</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>335</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.575907</td>
    </tr>
  </tbody>
</table>
</div>



## Using Navie Bayes Classifier
Here we use Navie Bayes Classifier to attempt to make perdictions based on input, however there is not enough data so it is always giving a false prediction (false postive, you're false incident, but you're not)
### Preprocess the text


```python
import nltk
from nltk.stem import WordNetLemmatizer
stopwords = nltk.corpus.stopwords.words('english')

# had to install this followed this guide:  https://stackoverflow.com/questions/13965823/resource-corpora-wordnet-not-found-on-heroku
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
#train_data

## CREDIT:  https://www.analyticsvidhya.com/blog/2021/09/creating-a-movie-reviews-classifier-using-tf-idf-in-python/

train_X_non = train_data['short_description']# + " " + train_data['description']   # '0' refers to the review text
train_y = train_data['FalseIncident']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = incidents['short_description']# + " " + incidents['description']
test_y = incidents['FalseIncident']
train_X=[]
test_X=[]

def processText(text, i=0):
    processed_text = re.sub('[^a-zA-Z]', ' ', text[i])
    processed_text = processed_text.lower()
    processed_text = processed_text.split()
    processed_text = [lemmatizer.lemmatize(word) for word in processed_text if not word in set(stopwords)]
    processed_text = [' '.join(processed_text)]
    return processed_text

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
    
print(train_X[3])
```

    [nltk_data] Downloading package stopwords to /home/strick/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


    access dhcp reservation anythong dhcp



```python

#tf idf
tf_idf = TfidfVectorizer()
#applying tf idf to training data
X_train_tf = tf_idf.fit_transform(train_X)
#applying tf idf to training data
X_train_tf = tf_idf.transform(train_X)

print("n_samples: %d, n_features: %d" % X_train_tf.shape)

#transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(test_X)
print("n_samples: %d, n_features: %d" % X_test_tf.shape)

```

    n_samples: 6, n_features: 40
    n_samples: 355, n_features: 40


### Run algo


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# REF:  https://builtin.com/data-science/precision-and-recall
#naive bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)
#predicted y
y_pred = naive_bayes_classifier.predict(X_test_tf)

#Prediction is complete. Now, we print the classification report.

print(metrics.classification_report(test_y, y_pred, target_names=['FalseIncident', 'NotFalse']))
```

                   precision    recall  f1-score   support
    
    FalseIncident       0.00      0.00      0.00       349
         NotFalse       0.02      1.00      0.03         6
    
         accuracy                           0.02       355
        macro avg       0.01      0.50      0.02       355
     weighted avg       0.00      0.02      0.00       355
    


    /home/strick/.local/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/strick/.local/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/strick/.local/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



```python
print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, y_pred))
```

    Confusion matrix:
    [[  0 349]
     [  0   6]]


Based on this confusion matrix, my model is not predicting very well.   There are 0 instances where i predicted a postive outcome to be true, 6 instances whre i predicted an incident not to be false, but it was and 349 (all others) where i said it was a false incident but it's not!


```python
# Lets do a prediction
test = ["avengers are here to stay in the world of us"]


review = re.sub('[^a-zA-Z]', ' ', test[0])
review = review.lower()
review = review.split()
review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
test_processed =[ ' '.join(review)]

#test_processed = processText("This is unlike any kind of adventure movie my eyes")
test_processed
```




    ['avenger stay world u']




```python
test_input = tf_idf.transform(test_processed)
test_input.shape
```




    (1, 40)




```python
res=naive_bayes_classifier.predict(test_input)[0]
res
```




    'true'



## Clustering Similarity Scores (NO), cluster based on full insciet for unsupervised learning


```python
from sklearn.cluster import KMeans

# Convert the similarity scores to a numpy array
similarity_scores_array = similarity_scores.values.reshape(-1, 1)
#similarity_scores_array = similarity_scores.dropna().values.reshape(-1, 1)
similarity_scores_array


def simplify_category(df):
    df['category']=pd.get_dummies(df.category).drop('Software',axis=1)
    return df

def simplify(df, col_name, col, value):
    df[col_name]=pd.get_dummies(col).drop(value,axis=1)
    return df
```


```python

tmp = simplify_category(incidents)
def drop_features(df):
    return df.drop(['short_description', 'description', 'sys_created_by', 'u_inc_dept', 'sys_created_on', 'closed_at', 'comments_and_work_notes', 'similar_incidents'], axis=1)

#tmp = drop_features(incidents)

tmp = simplify(tmp, 'quickly_closed', tmp.quickly_closed, 'false')
tmp = simplify(tmp, 'FalseIncident', tmp.FalseIncident, 'False')

```


```python
# Initialize a k-means object with the desired number of clusters
k = 2
kmeans = KMeans(n_clusters=k, init='k-means++')

# Fit the k-means model to the similarity scores
kmeans.fit(X_test_tf)

# Get the cluster assignments for each similarity score
#cluster_labels = kmeans.labels_
cluster_labels = kmeans.fit_predict(X_test_tf)

print(cluster_labels)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0
     0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1 0 0 0 0 1 1
     0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0
     0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]


    /home/strick/.local/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(



```python
import matplotlib.pyplot as plt
 
#filter rows of original data
filtered_label0 = X_test_tf[cluster_labels == 0]
filtered_label0

filtered_label0 = filtered_label0.toarray()
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])

#plotting the results
plt.scatter(filtered_label0[:,1] , filtered_label0[:,1])
#plt.show()
```




    <matplotlib.collections.PathCollection at 0x7fc1dd3b9090>




    
![png](./output_47_1.png)
    



```python

```
