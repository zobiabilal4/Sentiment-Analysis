import pandas as pd
from transformers import pipeline, RobertaTokenizer
#Assigning labels to our web scrapped amazon product reviews
# Load the dataset
file_path = 'Amazon_reviews_data.csv'
df = pd.read_csv(file_path)

# Load the pre-trained model and tokenizer from Hugging Face
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')

# Function to classify the sentiment using RoBERTa
def get_sentiment_label(text):
    try:
        text = str(text)
        max_length = 512
        
        # Tokenize and truncate text
        tokens = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        
        # Run sentiment analysis
        result = sentiment_analyzer(text)
        label = result[0]['label']
        
        # Map label to sentiment
        if label == 'LABEL_0':
            return 'negative'
        elif label == 'LABEL_1':
            return 'neutral'
        elif label == 'LABEL_2':
            return 'positive'
        else:
            return 'neutral'
    except Exception as e:
        print(f"Error processing text: {text[:100]}... - {e}")
        return 'neutral'

# Combine 'Review' and 'Description' columns for sentiment analysis
#df['Combined_Text'] = df['Review'].fillna('').astype(str) 
# Apply the function to the 'Review' column
df['Labels'] = df['Review'].apply(get_sentiment_label)

# Save the updated dataframe to a new CSV
output_file_path = 'Labeled-Amazon-Reviews-Dataset.csv'
df.to_csv(output_file_path, index=False)

print("Labels assigned and saved to new CSV file.")
