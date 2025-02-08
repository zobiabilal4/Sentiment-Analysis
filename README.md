# Sentiment-Analysis
In this repository we will be 
## EXPLORING THE POWER OF SENTIMENT ANALYSIS: Transforming Amazon Reviews to Improve Customer Satisfaction and Make Informed Decisions

# Overview
- Machine learning project
- Sentimental analysis based on customer reviews
- Classification model
- Goal: Determine sentiment as positive, negative, or neutral

# Problem Statement
In the realm of Amazon's e-commerce platform, managing the influx of product reviews poses a challenge. This project aims to develop a sentiment analysis model tailored for Amazon reviews. Its goal is to accurately classify sentiments as positive, negative, or neutral, empowering sellers with actionable insights.Challenges include diverse product categories and varying review lengths. Success will enable informed decision-making and enhanced customer satisfaction.

# Methdology
We will generate our own dataset by applying the technique of web scraping on Amazon. For that purpose we will use Selenium. For text classification, we will take the help of TextBlob or any suitable platform.

# Business Scope
Our project involves designing a robust model that generalizes well across
different product categories, mitigating biases, and addressing potential
challenges such as varying review lengths and language nuances. The
successful implementation of this sentiment analysis model will empower
businesses to make data-driven decisions based on a comprehensive
understanding of customer sentiments expressed in online product reviews.
Moreover our model adds a powerful visualization to the product purchase.
We can get idea about the purchasing impact of the products also.
- Interactive Dashboards:
       Visual representation of sentiment trends for intuitive insights.
- Competitor Benchmarking:
      Comparative visualizations for competitive benchmarking.
- Geographical Insights:
      Regional sentiment mapping for targeted marketing strategies.
- Product Performance Metrics:
       Holistic visual metrics summarizing product performance.
- Response Impact Analysis:
       Visualizing the impact of seller responses on sentiment trends.
- Dynamic Data Filters:
       Granular analysis through dynamic filters in visualizations.
- Forecasting Trends:
       Predictive visualizations for proactive decision-making

# Sentiment Analysis on Web-Scraped Data

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Project Workflow](#project-workflow)
  1. [Web Scraping](#web-scraping)
  2. [Data Labeling](#data-labeling)
  3. [NLP Techniques for Sentiment Analysis](#nlp-techniques-for-sentiment-analysis)
  4. [Data Visualization](#data-visualization)
  5. [Frontend with Streamlit](#frontend-with-streamlit)
- [Future Scope](#future-scope)
- [Team & Acknowledgements](#team--acknowledgements)

## Introduction
This project is aimed at analyzing sentiment (positive, negative, or neutral) from web-scraped data using Natural Language Processing (NLP) techniques. The project workflow includes the following steps:

- Web scraping to gather data.
- Labeling the web-scraped data for sentiment analysis.
- Applying NLP techniques to build a sentiment analysis model.
- Visualizing the data and results.
- Creating a frontend using Streamlit for an interactive user experience.

## Getting Started
To get started with the project, follow the instructions in `requirements.txt`. Ensure you have all the required dependencies installed.

## Project Workflow

### 1. Web Scraping
The first step in this project is to gather data through web scraping. We used Python libraries such as Selenium to scrape data from various websites. The scraped data includes:

- Product Name
- Price
- Availability
- Brand
- Description
- Category
- Rating
- Review
- Star
- Date

For different products, reviews were gathered (about 14 to 15 reviews per product). This data can be found in the CSV file `Amazon_reviews_dataset.csv`. Web scraping code can be found in `web_scraping.ipynb`.

### 2. Data Labeling
Once the data was scraped, the next step was to label it. The labeling process involved categorizing each piece of text as positive, negative, or neutral. This labeled dataset is essential for training and evaluating our sentiment analysis model.

Labeling was done using two methods:
1. **TextBlob**
2. **Pre-trained RoBERTa model from Hugging Face**

We used RoBERTa-labeled data. However, you can also find TextBlob-labeled data in the `text_blob` folder. The code for labeling using RoBERTa is in `label_roberta.py`, and the labeled dataset is named `Labeled-Amazon-Reviews-Dataset.csv`.

### 3. NLP Techniques for Sentiment Analysis
With the labeled dataset `Labeled-Amazon-Reviews-Dataset.csv`, we applied various NLP techniques to preprocess the data, including:
- Tokenization
- Lemmatization
- Stop-word removal

As our negative labels were very few compared to positive and neutral ones, we used oversampling and undersampling techniques. We then built and trained a sentiment analysis model using the following algorithms:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- LightGBM
- AdaBoost

We selected **LightGBM** as our final model based on evaluation measures. LightGBM is a gradient boosting framework that uses tree-based learning algorithms. 

- The Jupyter Notebook containing preprocessing, model building, and evaluation measures is `nlp_sentiment_analysis_model.ipynb`.
- The pickled model can be found in `model_LGBM.pkl`.
- `tfidf_vectorizer.pkl` will be used when making predictions through the frontend.

### 4. Data Visualization
To better understand the data and the results of our sentiment analysis, we created visualizations using libraries such as **Matplotlib** and **Seaborn**. These visualizations include:
- Rating distribution
- Sentiment distribution plots

The dataset `Labeled-Amazon-Reviews-Dataset.csv` was modified to `modified_file.csv`, which is a cleaner version and will be used in frontend visualization. Visualization code and results can be found in `Amazon_Reviews_Visualizations.ipynb`.

### 5. Frontend with Streamlit
Finally, we built a frontend for our project using **Streamlit**. The Streamlit app allows users (sellers of products) to:
- Interact with the sentiment analysis model.
- View visualizations.
- Input their own text to see the sentiment predictions.

This makes our project accessible and user-friendly. The frontend code is in `streamlit_app.py`.

Here is a preview of the frontend. For complete frontend pictures and videos, visit the `frontend_pics&video` folder.
![Frontend Preview](https://github.com/zobiabilal4/Sentiment-Analysis/blob/main/frontend_pics%26video/frt1.png)


## Future Scope
This project can be improved and enhanced on a much larger scale. Some suggestions include:

- Using a larger dataset with a fair representation of all sentiments.
- Implementing better models.
- Expanding to include product buyers' perspectives.

## Team & Acknowledgements
### Team Members:
- **Eman Zahid** (BSDSF21A010) - Data Science student at PUCIT, Lahore.
- **Zobia Bilal** (BSDSF21A026) - Data Science student at PUCIT, Lahore.

We are cordially thankful to our teacher and all the people who guided us in this project.

---

🎯 **Contributions are welcome!** If you find any issues or have suggestions, feel free to open an issue or a pull request. 🚀
