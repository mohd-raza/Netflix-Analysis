import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# For modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# For word frequency analysis
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Make sure stopwords are downloaded
nltk.download('stopwords')

# -----------------------------------------
#  Streamlit Configuration & Caching
# -----------------------------------------

st.set_page_config(page_title="Netflix Data Analysis", layout="wide")

@st.cache_data
def load_data():
    """Load the Netflix dataset from CSV and do initial cleaning."""
    df = pd.read_csv("netflix_titles.csv")
    
    # Impute missing values
    columns_to_impute = ['director', 'country', 'cast', 'rating', 'duration']
    for col in columns_to_impute:
        df[col] = df[col].fillna("Unknown")
    
    # Convert 'date_added' to datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    return df

# Global load
netflix_data = load_data()

# -----------------------------------------
#  Helper functions
# -----------------------------------------

def page_home():
    """Home Page / Introduction"""
    st.title("Welcome to the Netflix Data Analysis Dashboard")
    st.markdown(
        """
        This **multi-page** Streamlit application demonstrates an **end-to-end** analysis of 
        the Netflix dataset, including:  
        - **Data Overview & Cleaning**  
        - **Basic Exploratory Data Analysis (EDA)**  
        - **Advanced EDA** (Deeper Insights)  
        - **Predictive Modeling** (Random Forest)  
        - **Interactive Prediction**  

        Use the sidebar to navigate between pages.
        """
    )
    with st.expander("Preview Dataset (first 5 rows)"):
        st.dataframe(netflix_data.head())

def page_data_overview_cleaning():
    """Page: Data Overview & Missing Values"""
    st.header("Data Overview & Cleaning")
    
    # 1) Data dimensions
    st.write("**Dataset Dimensions:**", netflix_data.shape)
    
    # 2) Column dtypes
    st.subheader("Column Data Types")
    dtypes_df = pd.DataFrame(netflix_data.dtypes, columns=["Data Type"])
    st.dataframe(dtypes_df)
    
    # 3) Missing values analysis
    st.subheader("Missing Values Analysis")
    null_counts = netflix_data.isnull().sum()
    total_rows = len(netflix_data)
    null_percentage = (null_counts / total_rows) * 100
    null_df = pd.DataFrame({
        'Column': null_counts.index,
        'MissingCount': null_counts.values,
        'Missing%': null_percentage.values
    }).sort_values(by='Missing%', ascending=False)
    st.dataframe(null_df)
    
    st.markdown(
        """
        **Note**: Missing values in columns ('director', 'country', 'cast', 'rating', 'duration') 
        are imputed with 'Unknown'. Also, 'date_added' is converted to a datetime object. 
        """
    )

def page_eda_basic():
    """Page: Basic EDA (Distribution, top categories, etc.)"""
    st.header("Basic Exploratory Data Analysis (EDA)")
    
    # A) Distribution of Content Type
    st.subheader("Distribution of Content Types (Movie vs. TV Show)")
    type_counts = netflix_data['type'].value_counts().reset_index()
    type_counts.columns = ['type', 'count']
    fig_type_pie = px.pie(type_counts, names='type', values='count', title="Content Type Distribution")
    st.plotly_chart(fig_type_pie, use_container_width=True)
    
    # B) Top 5 Genres
    st.subheader("Top 5 Genres")
    # Split the 'listed_in' column
    genre_counts = netflix_data['listed_in'].str.split(', ').explode().value_counts()
    top_5_genres = genre_counts.head(5).reset_index()
    top_5_genres.columns = ['Genre', 'Count']
    fig_top_genres = px.bar(top_5_genres, x='Genre', y='Count', title="Top 5 Genres", color='Genre')
    st.plotly_chart(fig_top_genres, use_container_width=True)
    
    # C) Top 5 Actors
    st.subheader("Top 5 Actors")
    cast_counts = netflix_data['cast'].str.split(', ').explode().value_counts()
    top_5_cast = cast_counts.head(5).reset_index()
    top_5_cast.columns = ['Actor', 'Appearances']
    fig_top_actors = px.bar(top_5_cast, x='Actor', y='Appearances', title="Top 5 Actors", color='Actor')
    st.plotly_chart(fig_top_actors, use_container_width=True)
    
    # D) Release Year Distribution
    st.subheader("Release Year Distribution")
    fig_release_year = px.histogram(netflix_data, x="release_year", nbins=50, title="Release Year Distribution", color="release_year")
    st.plotly_chart(fig_release_year, use_container_width=True)
    
    # E) Content Rating Distribution
    st.subheader("Content Rating Distribution")
    fig_rating = px.histogram(netflix_data, x="rating", title="Rating Distribution", color="rating")
    st.plotly_chart(fig_rating, use_container_width=True)

def page_eda_advanced():
    """Page: Advanced EDA (Duration vs Rating, Comedies Analysis, Word Frequencies, etc.)"""
    st.header("Advanced EDA & Deeper Insights")

    # 1) Titles Added Over Time
    st.subheader("Titles Added Over Time (Year Added)")
    df_time = netflix_data.dropna(subset=['date_added']).copy()
    df_time['year_added'] = df_time['date_added'].dt.year
    year_counts = df_time['year_added'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax, palette="viridis")
    ax.set_title("Number of Titles Added per Year", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year Added")
    ax.set_ylabel("Count of Titles")
    st.pyplot(fig)

    # 2) Movie Duration vs. Rating (Boxplot)
    st.subheader("Movie Duration by Rating")
    movies_df = netflix_data[netflix_data['type'] == 'Movie'].copy()
    # Convert duration (e.g., "103 min") to numeric
    movies_df['duration_minutes'] = movies_df['duration'].str.replace(' min', '', regex=False)
    movies_df['duration_minutes'] = pd.to_numeric(movies_df['duration_minutes'], errors='coerce')
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=movies_df, 
        x='rating', 
        y='duration_minutes', 
        showfliers=False, 
        palette='Set2',
        ax=ax2
    )
    ax2.set_title("Distribution of Movie Durations by Rating", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Duration (minutes)")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # 3) Comedies Over the Years by Country
    st.subheader("Comedies Over the Years by Country (Top 10)")
    comedies_df = netflix_data[netflix_data['listed_in'].str.contains("Comedies", na=False)].copy()
    comedies_df['release_year'] = pd.to_numeric(comedies_df['release_year'], errors='coerce')
    comedies_expanded = comedies_df.assign(country=comedies_df['country'].str.split(', ')).explode('country')
    top_10_countries = comedies_expanded['country'].value_counts().head(10).index
    comedies_expanded_top10 = comedies_expanded[comedies_expanded['country'].isin(top_10_countries)]
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.countplot(
        data=comedies_expanded_top10,
        x='release_year',
        hue='country',
        order=sorted(comedies_expanded_top10['release_year'].unique()),
        ax=ax3
    )
    ax3.set_title("Comedies by Release Year (Top 10 Producing Countries)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Release Year")
    ax3.set_ylabel("Count")
    ax3.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig3)

    # 4) Word Frequency in Descriptions (Focus on Comedies)
    st.subheader("Word Frequency in Comedies' Descriptions")
    all_comedy_desc = " ".join(comedies_df['description'].astype(str)).lower()
    all_comedy_desc_clean = re.sub(r'[^\w\s]', '', all_comedy_desc)  # remove punctuation
    words = all_comedy_desc_clean.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    word_freq = Counter(filtered_words).most_common(15)
    common_words_df = pd.DataFrame(word_freq, columns=['Word', 'Count'])
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=common_words_df, x='Count', y='Word', palette='Reds_r', ax=ax4)
    ax4.set_title("Top 15 Keywords in Comedies' Descriptions", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Frequency")
    ax4.set_ylabel("Keyword")
    st.pyplot(fig4)

def page_predictive_modeling():
    """Page: Predictive Modeling (Random Forest), including user input for on-the-fly predictions."""
    st.header("Predictive Modeling: Random Forest (Movie vs. TV Show)")

    # Make a copy so we don't overwrite the original
    df_model = netflix_data.copy()

    # Drop datetime columns or any columns you do NOT want for modeling
    if 'date_added' in df_model.columns:
        df_model.drop('date_added', axis=1, inplace=True)

    # If you also created 'year_added' as an integer, you can keep it.
    # But if it's still datetime or problematic, drop it too:
    # if 'year_added' in df_model.columns:
    #     df_model.drop('year_added', axis=1, inplace=True)

    # Label encode all object columns (excluding 'type' which we do separately)
    object_cols = df_model.select_dtypes(include=['object']).columns
    encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    # Separate features (X) and target (y)
    # NOTE: 'type' should already be label-encoded in the step above
    X = df_model.drop(["type"], axis=1)
    y = df_model["type"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.write(f"**Random Forest Classifier Accuracy:** {accuracy:.2f}")

    st.markdown("---")
    st.subheader("Predict a Single Title (Movie or TV Show)")
    st.markdown("Use the form below to input features and predict content type.")

    # Create input form for each feature in X
    user_input = {}
    feature_cols = list(X.columns)

    for col in feature_cols:
        if col in encoders:
            # It's categorical
            example_val = encoders[col].classes_[0]
            val_str = st.text_input(f"Enter a value for '{col}' (categorical)", value=str(example_val))
            user_input[col] = val_str
        else:
            # It's numeric
            median_val = float(X[col].median())
            val_num = st.number_input(f"Enter a numeric value for '{col}'", value=median_val)
            user_input[col] = val_num

    if st.button("Predict Content Type"):
        # Convert the user inputs into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Apply label encoders to categorical fields
        for col in input_df.columns:
            if col in encoders:
                encoder = encoders[col]
                user_val = input_df[col].iloc[0]
                if user_val not in encoder.classes_:
                    st.warning(f"'{user_val}' not seen in training data; using default category.")
                    input_df[col] = encoder.transform([encoder.classes_[0]])
                else:
                    input_df[col] = encoder.transform(input_df[col])

        # Convert to numeric
        input_df = input_df.astype(float)

        # Predict
        pred_class = rf.predict(input_df)[0]
        # Convert numeric prediction back to original label
        type_encoder = encoders['type']
        pred_label = type_encoder.inverse_transform([pred_class])[0]

        st.success(f"**Predicted Content Type:** {pred_label}")


# -----------------------------------------
#  App Navigation
# -----------------------------------------

def main():
    st.sidebar.title("Netflix Dashboard Navigation")
    pages = {
        "Home": page_home,
        "Data Overview & Cleaning": page_data_overview_cleaning,
        "Basic EDA": page_eda_basic,
        "Advanced EDA": page_eda_advanced,
        "Predictive Modeling": page_predictive_modeling
    }
    
    choice = st.sidebar.selectbox("Select a page", list(pages.keys()))
    pages[choice]()  # Render the chosen page

if __name__ == "__main__":
    main()
