# app.py

from flask import Flask, render_template, request
from googleapiclient.discovery import build
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend suitable for web apps
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Initialize YouTube API
API_KEY = 'AIzaSyAkGNZPX7RA9Gm9H5Bg8kpqu17Gc8yFxGQ'  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Function to get comments from a video
# Function to get comments from a video with a total limit
def get_comments(video_id, max_results=100, total_limit=2000):
    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results,
        textFormat='plainText'
    ).execute()

    while response and len(comments) < total_limit:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            
            # Stop if we've reached the total limit
            if len(comments) >= total_limit:
                break
        
        # Check if there are more comments
        if 'nextPageToken' in response and len(comments) < total_limit:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=max_results,
                textFormat='plainText'
            ).execute()
        else:
            break

    return comments

# Data cleaning function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Sentiment analysis function using VADER
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 1  # Positive
    elif score['compound'] <= -0.05:
        return 0  # Negative
    else:
        return -1  # Neutral

# Generate sentiment analysis and visualizations
def analyze_comments(video_id):
    try:
        # Get comments
        comments = get_comments(video_id, max_results=100, total_limit=2000)
        if not comments:
            return None

        data = pd.DataFrame(comments, columns=['comment_text'])
        data['cleaned_text'] = data['comment_text'].apply(clean_text)
        data['sentiment'] = data['cleaned_text'].apply(get_sentiment)

        # Remove neutral comments
        data = data[data['sentiment'] != -1]

        # Feature extraction using TF-IDF
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(data['cleaned_text']).toarray()
        y = data['sentiment']

        # Train the Logistic Regression model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Sentiment Breakdown
        positive_comments = data[data['sentiment'] == 1]
        negative_comments = data[data['sentiment'] == 0]
        total_comments = len(data)
        positive_percentage = (len(positive_comments) / total_comments) * 100
        negative_percentage = (len(negative_comments) / total_comments) * 100

        # Generate Pie Chart
        plt.figure(figsize=(8, 5))
        labels = ['Positive', 'Negative']
        sizes = [positive_percentage, negative_percentage]
        colors = ['#66b3ff','#ff9999']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title('Sentiment Breakdown')

        # Save pie chart to a string
        pie_buffer = BytesIO()
        plt.savefig(pie_buffer, format="png")
        pie_buffer.seek(0)
        pie_image = base64.b64encode(pie_buffer.getvalue()).decode()
        plt.close()

        # Generate Word Clouds
        positive_text = ' '.join(positive_comments['cleaned_text'].tolist())
        negative_text = ' '.join(negative_comments['cleaned_text'].tolist())

        wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        wordcloud_negative = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)

        # Save word cloud images to strings
        wc_positive_buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_positive, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Positive Comments')
        plt.savefig(wc_positive_buffer, format="png")
        wc_positive_buffer.seek(0)
        positive_image = base64.b64encode(wc_positive_buffer.getvalue()).decode()
        plt.close()

        wc_negative_buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_negative, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Negative Comments')
        plt.savefig(wc_negative_buffer, format="png")
        wc_negative_buffer.seek(0)
        negative_image = base64.b64encode(wc_negative_buffer.getvalue()).decode()
        plt.close()

        return {
            'pie_chart': pie_image,
            'positive_wordcloud': positive_image,
            'negative_wordcloud': negative_image,
            'positive_percentage': positive_percentage,
            'negative_percentage': negative_percentage,
            'total_comments': total_comments
        }
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['video_url']
        # Extract video ID from YouTube URL
        if 'v=' in video_url:
            video_id = video_url.split('v=')[-1]
        else:
            video_id = video_url.split('/')[-1]
        
        results = analyze_comments(video_id)
        if results:
            return render_template('index.html', results=results)
        else:
            return render_template('index.html', error="Could not retrieve or analyze comments.")

    return render_template('index.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
