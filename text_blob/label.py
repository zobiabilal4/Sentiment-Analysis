import pandas as pd
from textblob import TextBlob

# Load the dataset
file_path = 'Amazon_reviews_dataset.csv'
df = pd.read_csv(file_path)

# Function to classify the sentiment
def get_sentiment_label(review):
    if isinstance(review, float):
        review = ''  # Convert NaN or non-string values to empty string
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0.1:
        return 'positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# Apply the function to the 'Review' column
df['Labels'] = df['Review'].apply(get_sentiment_label)

# Save the updated dataframe to a new CSV
output_file_path = 'Labeled_Amazon_reviews_dataset.csv'
df.to_csv(output_file_path, index=False)

print("Labels assigned and saved to new CSV file.")
