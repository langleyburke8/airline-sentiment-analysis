**Airline Reviews is the original data- will have to go through all cleaning and geoply to get coordinates, or can load "updated_Airline_review1.csv" and use that to run smoothly and quickly through the notebooks (processed and cleaned data)**

**Implementation Instructions:**
1. Download the Zip File:

Download the zip file containing all the necessary files (this will include Python scripts, notebooks, data files, models, and any other dependencies).
Extract the Files:

2. Extract the contents of the zip file to a folder on your local machine or server.
Upload the Folder to a Python-based Environment:

If you're working in a cloud environment (like Jupyter Notebook on a server or cloud service), upload the folder containing all extracted files to your environment.
In your local Python environment, make sure the folder is accessible.
Set the Working Directory to the folder if using. 

3. In your Python environment (e.g., Jupyter Notebook), set the working directory to the folder containing the files. You can do this using the following code:

import os
os.chdir('path_to_folder')  # Replace 'path_to_folder' with the actual path

This ensures that your script can access all the necessary files in the folder.

4. Run the app.ipynb File:

Open the app.ipynb Jupyter notebook file within the environment. This file will:
        Load the clean data.
        Load the saved API token and model.
        Execute the cells in the notebook to run the application.


## Intro
The report will discuss where data was found, the type and eda, the cleaning that was done, the model that was created, the platform and business value for the app creation and lastly the dashboard overview with the final insights and improvements/limitations.

This project uses data from Kaggle (https://www.kaggle.com/datasets/juhibhojani/airline-reviews), which is a collection of Airline Reviews and it is used to classify the reviews as positive - hence leading to a recommendation to others or negative and therefore no recommendation. The data will also be used to conduct a sentiment analysis for an airline to gain insight into the publics opinions. The model, as mentioned, is a `Keras` classification model that uses the review text data and predicts whether the customer will recommend the airline or not. The reason behind building this model is to allow airlines to get insights into not only the publics view on themselves but also to their competitors based on this publicly accessible data. The idea is that the companies can determine where they can improve based on customer feedback, and it also will give insight on busiest routes and overall ratings for those routes. Then finally being able to enter in customer feedback to the trained classification model and getting the prediction of recommendation or not. In order to train the model the review data had to go through preprocessing and tokenization steps as well as a few others that will be described in the following section. To be able to conduct the sentiment analysis and populate the dashboards on the `Streamlit` web app the data had to go through cleaning steps as well - this included splitting 'Route' into the cities customers were flying to and from, which then were cleaned to be in Title format, to group city names that were slightly differing, for example "Houston Texas" instead of "Texas". Aside from those steps there were a few others that will be described in more detail later on. The dashboards created include - distributions of flights to and from specific cities, average ratings for those flights, whether customers are recommending based on them and a look into the average rating for customer satisfaction metrics such as wifi, entertainment, food and beverages, seat comfort and more. Flight maps are included as well as the text entry of reviews for prediction.

The data used and the cleaning .ipnyb file will be provided to enable replication. 


## Data

The data from Kaggle included the following columns:

- **Airline Name:** The airline being reviewed.
- **Overall Rating:** The user's overall rating for their experience (1-10).
- **Review Title:** The customers review title.
- **Review Date:** The date the review was submitted.
- **Verified:** Indicates if the review is from a verified user.
- **Review:** A detailed account of the user's experience (Used in model).
- **Aircraft:** The type of aircraft used on the flight.
- **Type of Traveller:** The category of the traveler (business, solo leisure, family leisure, couple leisure).
- **Seat Type:** The class of the seat reviewed (economy class, business class, premium economy class, first class).
- **Route:** The flight route taken by the user.
- **Date Flown:** The date the user took the flight.
- **Recommended:** Indicates whether the user recommends the airline (Used as sentiment in the model training).


Satisfaction Metrics:
- **Seat Comfort:** Rating of the comfort of the seat (1-5).
- **Cabin Staff Service:** Rating of the cabin crew's service (1-5).
- **Food & Beverages:** Rating of the onboard food and beverages (1-5).
- **Ground Service:** Rating of services provided on the ground (e.g., check-in, boarding) (1-5).
- **Inflight Entertainment:** Rating of onboard entertainment options (1-5).
- **Wifi & Connectivity:** Rating of the internet and connectivity services onboard (1-5).
- **Value for Money:** Rating of the overall cost-effectiveness of the flight (1-5).

Summary stats were looked at including descriptive summary of the dataset and data types, the most common words in the 'Review' column, the min, max and average of review length as well as the sentiment counts (which will be commented on in the limitations) and the review formats. By exploring the data the issues that had to be fixed and some key metrix like the average review length and class distribution were identified.



## Cleaning:

There were several steps in cleaning and transforming the data and they will be listed below with explanation.

- **Splitting 'Route':** The 'Route' column was split so that the to and from city were seperated and new columns 'to_city' and 'from_city' were created. This way we could get coordinates and analyse patterns based on where customers were flying from and were they were going.

The city names were not grouping together successfully so the formatting and naming was altered. 

- **Formatting:** The 'to_city' and 'from_city' were converted to have the names in Title format. The accents were removed from the names - whether put there purposefully or by mistake. This would help make sure more city names were recognized to be the same. Empty reviews were also dropped from the data set.

- **Invalid Cities:** Some city names were entered and they were not valid - this gave warnings when extracting coordinates. The rows that contained these values were removed.

- **Renaming Observations:** The cleaning continued with converting column observations, specifically 'Recommended' was altered so 'yes' and 'no' were capitalized. The largest observation cleaning came when discovering differences in how the city names were entered for each review. In order to be grouped properly for observations, the city names that were similr but showing up as two seperate for the dashboards in the Streamlit app were converted to just the city name, for example "Orlando Airport" was changed to "Orlando". The logic that was used was any city name with "Orlanda" in it would be converted to just that- this was used for many other city names. Another issue that was discovered was the use of airport codes as city names. In order to deal with this another dataset was loaded, one that contained airport codes and city names. The airport codes were matched to those in the Airline Review dataset and the city names associated replaced the codes. This dataset can be found here: https://data.opendatasoft.com/explore/dataset/airports-code%40public/table/?flg=en-us. 

- **Coordinates:** In order to populate a flight map on the Streamlit app the coordinates were needed for all cities. The library `geopy.geocoders` was used here, the Namoinatim interface for the Nominatim API was imported to impute coordinates into the dataset. As mentioned it could not locate the coordinates for a few cities - those with formatting issues, but majority were found. The columns 'from_coords', 'to_coords', 'from_lat', 'from_long', 'to_lat' and 'to_long' were added into the 'review_data' *This code chunk took 2 hours to run, for that reason it is commented out in the data cleaning and modeling notebook. When trying to replicate the work, the Updated_Airline.csv file can be used with the coordinates saved, or run all with original Kaggle file*

- **Recommended Numeric:** Added a column to encode the 'Recommended' column as 1 = 'Yes', else 0. This made some groupings for dashboards simpler.

- **Data Type:** The datatype for the variable 'Overall_Rating' had to be converted from a object to a float.

Although more cleaning could be done, the basic issues in the dataset were adressed given the time - this will be discussed later on as well. The data is now clean for use to begin the model training steps!


## Model

Model Steps: 

In the beggining of the model section, the data was explored but now with a focus on the sentiment (Recommended or not) and the reviews. The sentiment counts were found for each class- with negative overpowering positive cases with roughly 15300 and positive having only 7800 cases (this class imbalance can cause a training issue for the lack of understanding of what leads to positive cases due to less observations - this is examined further later on). The number of empty messages, maximum length of reviews and the average length were found (average length of 721 characters) - the average length of a review would be used in the vocabulary length step. Similarly the most common words were identified to be many stop words with one sticking out and that was "flight". 

**Steps for Building the Classification Model**

1. Preprocessing: The punctuation is removed from all reviews and the format becomes lowercase for all characters.

2. Generate Vocab to Index Mapping: Keras `Tokenizer` was used in this step to generate the word index, it takes the vocabulary defined and assigns each word a unique token from 1 to 23043 (The vocab size). Here, the tokenizer is saved to be brought into the `Streamlit` notebook.

3. Message Encoding and Labels: Here the text gets transformed to a number so the algorithm can take it as input.

4. Padding Messages: Padding messages so that the reviews are the same lengths for all inputs, using zero pad for the shorter reviews. The max length used is the average character length of 750 (raised higher - 721 is average).

5. Splitting the Data: Splitting the data into train and validation - ensuring that both splits have labels 0 and 1.

6. `GloVe` Embeddings: The pretrained `GloVe` embeddings will vectorize the data so that semantic meaning will be preserved moreso than without.

7. Model and Training: The model was then built using the ConvNet function in Keras and the architechure from the Yoon Kim model (https://arxiv.org/abs/1408.5882) with some adjustments like GlobalMaxPooling1D so that the model runs smoohtly in the `Streamlit` app. Training the model using 10 epochs due to time and machine limitations, we get a testing accuracy of 90% and a train of 99%. This suggest some overfitting but the data is relatively small so we would expect it to learn the trends within the train.

8. Saving the model for app implementation.


**Improvements/Limitations**

Given the time restraint there were areas I hoped to explore but did not get to, this includes a deep clean of the data - there are still some city names that are not formatted properly to be grouped together for the dashboards. Fixing this would give more reliable insights. Another thing would be to re run geoply after the city names were formatted properly to fill in the empty spaces that it could not find coordinates for. Thirdly, there is a class imbalance present in the dataset - the posisitve reviews are roughly half of the negative ones, by implementing a method like SMOTE this could better train the model to recognize a recommendation from review better. With that being said, because of a review sometimes being not overly positive or negetive the model struggles when there are both positive and negative aspects to conclude whether they would recommend or not. For this reason the class prediction percentages are shown in the app demo. This is also a reason to include the overall rating in the model training and convert this binary model to a multi classification, this way the model can take in all the views of each area of a flight and then predict what the rating they would give would be. As for extra dashboards, incorporating the  flight date to analyse any seasonality trends for satisfied flyers would be an interesting thing to look at as well as trends in the seasons for count of flights per month. The dataset also has roughly 23000 rows which makes it somewhat small- to improve dashboards, insights and model predictions a bigger datatset would be better for future implementation.