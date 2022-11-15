# Progress on cp clim indices notebook :

## 31/10/2022 :
- Started notebook
- studied correlation between cp and clim indices for europe as a whole and for different time resolution

## 03/11/2022
- study the corr for diff countries
- plot choropleith map based on corr
- Meeting with Wafa

## 06/11/2022
- Added catboost and CNN for cp regression 
- Added catboost for LWP classification 

## 08/11/2022
Tried : 
- Catboost/ XGboost
- with undersampling, and with oversampling
- CNN
- change window_size parameter. window size is the number of past values for each feature that we use as input. ex: if window_size=2, X.shape = [batch_size,window_size,n_features]
- add remove climate indices
- add capacity factor history to the input 
- use data of IE for GB do increase training set size

09/11/2022 
- Used LWP from other countries to classify LWP
- Used LWP from neighbours countries to classify LWP
- Meeting with naveen 
- Created dataset with combined cf
- created TLCC for wind, solar and combined 

15/11/2022 Morning
- Created classification model with only LWP events of 10 most correlated neighbors
- choropleth map of most correlated neighbors for each country
- Run the classification model and store results for each country

15/11/2022 Afternoon
- Added nao and ao to see difference of results --> nao and ao don't bring much to the model
- visualize feature importance 

## What to do :
- Run counterfactuals
- See feature importance for each country and how it compares to correlation coeff
- progressively add : Own LWP history

this is a test





