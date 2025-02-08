import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

#Stremlit frontend for EXPLORING THE POWER OF SENTIMENT ANALYSIS: Transforming Amazon Reviews to Improve Customer Satisfaction and Make Informed Decisions


#pickle_in = open("model_Ada.pkl","rb")
#Kindly take note of path in your pc. Also there are attached pics in frontend pics folder and also frontend_video.mp4 in case of any running issue
pickle_in = open(r"C:\Users\Dell\OneDrive\Documents\Machine_Learning\ml_project\model_LGBM.pkl", "rb")
vectorizer_path = open(r"C:\Users\Dell\OneDrive\Documents\Machine_Learning\ml_project\tfidf_vectorizer.pkl", "rb")

classifier=pickle.load(pickle_in)
vectorizer = pickle.load(vectorizer_path)
data = pd.read_csv(r'C:\Users\Dell\OneDrive\Documents\Machine_Learning\ml_project\modified_file.csv')

def show_home_page():
    st.title("Amazon Product Reviews Sentiment Analysis")
    #st.image("C:/Users/Dell/OneDrive/Documents/Machine_Learning/ml_project/Amazon-Logo-31.png", use_column_width=True)

    st.write("""
        ### Welcome to the Sentiment Analysis of Amazon Product Reviews!
        
        This application allows you to explore Amazon product reviews, visualize data insights, compare model performances, and predict the sentiment of new reviews using various machine learning models.

        **Navigate through the menu to start!**
    """)


#streamlit run c:\Users\Dell\OneDrive\Documents\Machine_Learning\ml_project\streamlit_app.py
#@app.route('/predict',methods=["Get"])
def predict_note_authentication(review):
    
    review_vector = vectorizer.transform([review])
   
    prediction=classifier.predict(review_vector)
    print(prediction)
    return prediction
def plot_product_data(product_df,product_name):
        
        st.header(f"{product_name}")
        st.write(f"Price: {product_df['Price'].iloc[0]}$")
        st.write(f"Rating: {product_df['Rating'].iloc[0]}")
        # Plotting the star distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(data=product_df, x='Star',color="green")
        plt.title(f'Star Distribution for {product_name}   ',color="green")
        plt.xlabel('Star Given by Customers',color="red")
        plt.ylabel('Count',color="red")
        
        # Plotting the sentiment distribution
        plt.subplot(1, 2, 2)
        sns.countplot(data=product_df, x='Labels',color="blue")
        plt.title(f'      Sentiment Distribution for {product_name}',color="blue")
        plt.xlabel("Sentiment of Reviews",color="red")
        plt.ylabel('Count',color="red")
        
        plt.tight_layout()
        return plt



def main():
    
    pages = ["Home","Sentiment Analysis", "Beauty & Personal Care","Clothing, Shoes & Jewelry","Office Products","Toys & Games","Sports & Outdoors", "About"]
    page = st.sidebar.radio("Menue", pages)
    #----------------------------------------------------------------------------------------
    if page == "Home":
      show_home_page()
    if page == "Sentiment Analysis":
      st.title("Amazon Reviews Sentiment Analysis")
      html_temp = """<div style="background-color:tomato;padding:10px">
      <h2 style="color:white;text-align:center;">Review Sentiment Analysis App for Sellers </h2>
      </div>"""
      st.markdown(html_temp,unsafe_allow_html=True)
      st.write("")
      review = st.text_input("Review","Welcome Sellers! Type Here......")
      result=""
      if st.button("Predict"):
          result=predict_note_authentication(review)
      st.success('The sentiment is {}'.format(result))
      if st.button("About"):

          st.text("This application helps you to analyze the sentiment of your reviews.") 
          st.text("Seniments can be positive, negative or neutral.")
    #----------------------------------------------------------------------------------------
    if page == "Beauty & Personal Care":
      st.title("Beauty and Personal Care")
          
    # Get the list of unique product names
      product_names = data['Name'].unique()
      product_category="Beauty & Personal Care"

      # Loop through each product and plot the data
      for product_name in product_names:
        product_df = data[(data['Name'] == product_name) & (data['Category'] == product_category)]
        if product_df.empty:
          continue
        else:
          fig=plot_product_data(product_df, product_name)
          
          st.pyplot(fig)
    #----------------------------------------------------------------------------------------

    if page == "Clothing, Shoes & Jewelry":
      st.title("Clothing, Shoes & Jewelry")
          
    # Get the list of unique product names
      product_names = data['Name'].unique()
      product_category="Clothing, Shoes & Jewelry"

      # Loop through each product and plot the data
      for product_name in product_names:
        product_df = data[(data['Name'] == product_name) & (data['Category'] == product_category)]
        if product_df.empty:
          continue
        else:
          fig=plot_product_data(data, product_name)
          st.pyplot(fig)
  #----------------------------------------------------------------------------------------

    if page == "Office Products":
      st.title("Office Products")
          
    # Get the list of unique product names
      product_names = data['Name'].unique()
      product_category="Office Products"

      # Loop through each product and plot the data
      for product_name in product_names:
        product_df = data[(data['Name'] == product_name) & (data['Category'] == product_category)]
        if product_df.empty:
          continue
        else:
          fig=plot_product_data(data, product_name)
          st.pyplot(fig)
  #----------------------------------------------------------------------------------------

    if page == "Toys & Games":
      st.title("Toys & Games")
          
    # Get the list of unique product names
      product_names = data['Name'].unique()
      product_category="Toys & Games"

      # Loop through each product and plot the data
      for product_name in product_names:
        product_df = data[(data['Name'] == product_name) & (data['Category'] == product_category)]
        if product_df.empty:
          continue
        else:
          fig=plot_product_data(data, product_name)
          st.pyplot(fig)
  #----------------------------------------------------------------------------------------

    if page == "Sports & Outdoors":
      st.title("Sports & Outdoors")
          
    # Get the list of unique product names
      product_names = data['Name'].unique()
      product_category="Sports & Outdoors"

      # Loop through each product and plot the data
      for product_name in product_names:
        product_df = data[(data['Name'] == product_name) & (data['Category'] == product_category)]
        if product_df.empty:
          continue
        else:
          fig=plot_product_data(data, product_name)
          st.pyplot(fig)

    #----------------------------------------------------------------------------------------
    if page == "About":
      st.title("About")
      st.write("""
        The application demonstrates the sentiment analysis of product reviews using the LightGBM model for the sentiment prediction.
        This application shows viualizations of amazon products of different categories.
        
        **Project Contributors:**
        - Eman Zahid
        - Zobia Bilal
        
        **Acknowledgments:**
        - Dataset: Amazon Product Reviews Scraped from Amazon Online Store
        - Libraries: Streamlit, scikit-learn, NLTK, LightGBM, and more.
    """)


    
if __name__=='__main__':
    main()


    