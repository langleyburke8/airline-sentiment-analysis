import re
import pandas as pd
import plotly.express as px
import streamlit as st
import pydeck as pdk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import utils as utl
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json



st.title("Airline Sentiment Analysis")

##LOad Data 
file = 'updated_Airline_review1.csv'  # Path to your CSV file
review_data = pd.read_csv(file)

##Filters 

# Sidebar: Filters for Airlines and Recommendations
airline_filter = st.sidebar.multiselect(
    'Select Airline(s)', 
    options=review_data['Airline Name'].unique(), 
    key="airline_filter"  
)

recommendation_filter = st.sidebar.multiselect(
    'Select Recommendation(s)', ['Yes', 'No'], 
    default=['Yes', 'No'], 
    key="recommendation_filter"
)


filtered_df = review_data[
    review_data['Airline Name'].isin(airline_filter) &
    review_data['Recommended'].isin(recommendation_filter) 
]

# Drop rows with null or empty values in 'from_city' or 'to_city'
filtered_df = filtered_df.dropna(subset=['from_city', 'to_city'])

# Alternatively, exclude rows where 'from_city' or 'to_city' are empty strings
filtered_df = filtered_df[
    (filtered_df['from_city'].str.strip() != '') &
    (filtered_df['to_city'].str.strip() != '')
]


## Flight Map 

mapbox_api_key = 'pk.eyJ1IjoibGFuZ2xleWJ1cmtlOCIsImEiOiJjbTRlb2twNTAweWg5MmxvdjJkcmh2NnQwIn0.zt1K4-2qgesEp6fk24eudA'  # Replace this with the copied token
pdk.settings.mapbox_api_key = mapbox_api_key

# Drop rows with missing coordinates or city data
filtered_df = filtered_df.dropna(subset=['from_lat', 'from_lon', 'to_lat', 'to_lon', 'from_city', 'to_city', 'Route', 'Airline Name'])

# Define the Pydeck Layer for flight paths with tooltips
arc_layer = pdk.Layer(
    "ArcLayer",
    data=filtered_df,
    get_source_position=["from_lon", "from_lat"],  # Start point
    get_target_position=["to_lon", "to_lat"],      # End point
    get_width=2,
    get_tilt=15,
    get_source_color=[0, 255, 0],  # Arc source color (green)
    get_target_color=[0, 140, 255],  # Arc target color (blue)
    pickable=True,
    auto_highlight=True,
)


# Render the map in Streamlit
st.pydeck_chart(
    pdk.Deck(
        layers=[arc_layer],
        tooltip = {"text": "Route: {Route} \nAirline Name: {Airline Name}"},
        map_style="mapbox://styles/mapbox/light-v9"  # You can change the map style here
    )
)


##Flight Distribution By Airline - Recommended or Not 

# Step 1: Distribution of Flights - Histograms for From/To City with Recommendation Legends
st.subheader("Distribution of Flights - From and To Cities")

# Create interactive histogram for From City with Recommendation Legends
fig_from_city2 = px.histogram(filtered_df, 
                             x='from_city', 
                             color='Recommended', 
                             title='Flights From Cities by Recommendation', 
                             labels={'from_city': 'City Passenger Flies From'},
                             barmode='group', 
                            facet_col="Airline Name",
                             category_orders={'Recommended': ['No', 'Yes']},
                             width=1000,  # Adjust the width
                            height=600 )

# Create interactive histogram for To City with Recommendation Legends
fig_to_city2 = px.histogram(filtered_df, 
                           x='to_city', 
                           color='Recommended', 
                             title='Flights To Cities by Recommendation',
                            labels={'to_city': 'City Passenger Fly To'},
                             barmode='group',
                            facet_col="Airline Name",
                             category_orders={'Recommended': ['No', 'Yes']},
                           width=1000,  # Adjust the width
                            height=600 )

# Display the interactive plots
st.plotly_chart(fig_from_city2)
st.plotly_chart(fig_to_city2)




## Tables for Most Popular Routes

# Step 1: Identify busiest FROM and TO cities
from_city_counts = filtered_df['from_city'].value_counts().reset_index()
from_city_counts.columns = ['from_city', 'flight_count']  # Rename columns for from_city

to_city_counts = filtered_df['to_city'].value_counts().reset_index()
to_city_counts.columns = ['to_city', 'flight_count']  # Rename columns for to_city

# Streamlit layout with two columns for side-by-side display
col1_From, col2_To = st.columns(2)

with col1_From:
    st.subheader("Busiest From Cities")
    st.write(from_city_counts)  # Display the from_city_counts DataFrame with new column names

with col2_To:
    st.subheader("Busiest To Cities")
    st.write(to_city_counts)


# Step 3: Visualize Busiest Cities with Bar Charts
bar_from_city = px.bar(from_city_counts, x='from_city', y='flight_count', 
                       title='Busiest From Cities', 
                       labels={'from_city': 'City', 'flight_count': 'Number of Flights'})

bar_to_city = px.bar(to_city_counts, x='to_city', y='flight_count', 
                     title='Busiest To Cities', 
                     labels={'to_city': 'City', 'flight_count': 'Number of Flights'})

# Streamlit layout with two columns for side-by-side maps
col1_busyfrom, col2_busyto = st.columns(2)

with col1_busyfrom : st.plotly_chart(bar_from_city)
    
with col2_busyto:st.plotly_chart(bar_to_city)





##Customer Summary 

st.subheader("Customer Satisfaction Summary")

# Define labels for features dynamically

summary_df = (
    filtered_df.groupby(['Type Of Traveller', 'Seat Type', 'Airline Name'], as_index=False)
    .agg({
        'Seat Comfort': 'mean',
        'Cabin Staff Service': 'mean',
        'Food & Beverages': 'mean',
        'Ground Service': 'mean',
        'Inflight Entertainment': 'mean',
        'Wifi & Connectivity': 'mean',
        'Value For Money': 'mean'
    })
)

# Melt the DataFrame to plot all features
melted_df = summary_df.melt(
    id_vars=['Type Of Traveller', 'Seat Type', 'Airline Name'],
    var_name="Feature",
    value_name="Average Score"
)

# Create the stacked bar chart with tooltips
customer_fig = px.bar(
    melted_df,
    x="Type Of Traveller",
    y="Average Score",
    color="Feature",
    barmode="stack",
    facet_col="Airline Name",
    category_orders={'Seat Type': ['Business Class', 'First Class' 'Economy Class', 'Premium Economy']},
    labels={
        "value": "Average Score",
        "Type Of Traveller": "Customer Type",
        "Seat Type": "Seat Class",
        "Feature": "Feature",
        "Airline Name": "Airline"
    },
    hover_data={
        "Average Score": ":.2f",  # Format the average score to 2 decimal places
        "Feature": True,
        "Seat Type": True,
        "Type Of Traveller": True,
        "Airline Name": True 
    },
    title="Customer Profile Summary by Features"
)

# Update facet column titles
customer_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # Keep only the value (e.g., 'Business')

# Show the chart
st.plotly_chart(customer_fig)


##Happy customers FROM cities

# Group the data by city and recommendation to calculate the count of recommendations and average rating
city_from_data = (
    filtered_df.groupby(['from_city', 'from_lat', 'from_lon', 'Recommended', 'Airline Name'], as_index=False)
    .agg({'Overall_Rating': 'mean', 'Recommended': 'count'})
    .rename(columns={'Overall_Rating': 'avg_rating', 'Recommended': 'count_of_recommendations'})
)

# Assign colors based on the average rating (Darker green = higher rating, lighter green = lower rating)
city_from_data['color'] = city_from_data['avg_rating'].apply(
    lambda x: [0, int(255 * (1 - (x - 5) / 5)), 0]  # Linear interpolation to get lighter or darker green
)

# Set up the Pydeck map layer
layer_from = pdk.Layer(
    'ScatterplotLayer',
    city_from_data,
    get_position=['from_lon', 'from_lat'],  # Longitude and latitude
    get_radius='count_of_recommendations',  # Circle size based on recommendation count
    get_fill_color='color',  # Color based on recommendation (Green, darker = higher rating)
    radius_min_pixels=5,
    radius_max_pixels=50,
    pickable=True,
    opacity=0.6
)

##Happy customers TO cities

city_to_data = (
    filtered_df.groupby(['to_city', 'to_lat', 'to_lon', 'Recommended', 'Airline Name'], as_index=False)
    .agg({'Overall_Rating': 'mean', 'Recommended': 'count'})
    .rename(columns={'Overall_Rating': 'avg_rating', 'Recommended': 'count_of_recommendations'})
)

# Assign colors based on the average rating (Darker green = higher rating, lighter green = lower rating)
city_to_data['color'] = city_to_data['avg_rating'].apply(
    lambda x: [0, int(255 * (1 - (x - 5) / 5)), 0]  # Linear interpolation to get lighter or darker green
)

# Set up the Pydeck map layer
layer_to = pdk.Layer(
    'ScatterplotLayer',
    city_to_data,
    get_position=['to_lon', 'to_lat'],  # Longitude and latitude
    get_radius='count_of_recommendations',  # Circle size based on recommendation count
    get_fill_color='color',  # Color based on recommendation (Green, darker = higher rating)
    radius_min_pixels=5,
    radius_max_pixels=50,
    pickable=True,
    opacity=0.6
)

# Streamlit layout with two columns for side-by-side maps
col1, col2 = st.columns(2)

with col1 :
    st.subheader("Cities Travelled From - Customer Ratings")

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer_from],
            tooltip = {"text" : "Average Rating: {avg_rating} \nAirline Name: {Airline Name} \nCity: {from_city}"}, 
            map_style="mapbox://styles/mapbox/light-v9"
        )
    )

    
with col2 :
    st.subheader("Map of Cities Travelled To - Customer Ratings")

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer_to],
            tooltip = {"text" : "Average Rating: {avg_rating} \nAirline Name: {Airline Name} \nCity: {to_city}"}, 
            map_style="mapbox://styles/mapbox/light-v9"
        )
    )




##Bias Analysis

st.subheader("Ratings Leading to Recommendations")

# Step 1: Identify Ratings and Recommendations
bias_analysis = filtered_df[['Overall_Rating', 'Recommended', 'Airline Name']]
bias_analysis.columns = ['Overall_Rating', 'Recommended', 'Airline Name']

# Step 2: Group and count data for the bar chart
bias_analysis_grouped = (
    bias_analysis
    .groupby(['Overall_Rating', 'Recommended', 'Airline Name'])
    .size()
    .reset_index(name='Count')
)

# Step 3: Create the bar chart
bar_bias_analysis = px.bar(
    bias_analysis_grouped,
    x='Overall_Rating',
    y='Count',
    color='Recommended',  # Different bars for each recommendation status
    title='Ratings Turned to Recommendations',
    labels={'Overall_Rating': 'Ratings', 'Count': 'Count of Recommendations' , 'Airline Name': 'Airline Name'},
    barmode='group',# Grouped bars for better comparison
    facet_col = 'Airline Name'
)

st.plotly_chart(bar_bias_analysis)




##Classification Model

# Load the trained model
model = load_model('model_new_trained2.h5')

# Load the JSON file containing the tokenizer
with open('tokenizer.json') as json_file:
    tokenizer_json = json.load(json_file)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Define the max sequence length used during training
max_sequence_length = 750

# Streamlit UI
st.title("Review Sentiment")
st.write("Enter a customer review and the model will predict if the passenger would recommend or not.")

# User input for the review
review = st.text_area("Enter the review:")

if st.button('Predict'):
    if review:
        # Step 1: Preprocess the review (ensure this function is correct)
        preprocessed_review = utl.preprocess_ST_message(review)
        
        # Step 2: Convert to sequences using the tokenizer
        sequences = tokenizer.texts_to_sequences([preprocessed_review])

        # Display intermediate results
        st.write(f"**Preprocessed review:** {preprocessed_review}")
        st.write(f"**Tokenized sequences:** {sequences}")

        # Step 3: Pad the sequences to the same length as during training
        review_data_new = pad_sequences(sequences, maxlen=max_sequence_length)
        
        # Step 4: Make a prediction
        yNew = model.predict(review_data_new)
        
        # Step 5: Get the predicted class (binary, so use np.argmax)
        preds_classes = np.argmax(yNew, axis=-1)
        
        # Step 6: Display the prediction
        if preds_classes[0] == 1:
            st.write("**Prediction:** Will Recommend")
        else:
            st.write("**Prediction:** Will Not Recommend")
        
        # Display raw model output
        st.write(f"**Prediction output (raw model output):** {yNew}")
    else:
        st.write("Please enter a review text.")

