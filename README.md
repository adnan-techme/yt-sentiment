# YouTube Sentiment Analysis Web App

This is a web application that performs sentiment analysis on YouTube video comments using a simple web interface. The app leverages the YouTube Data API to fetch comments and uses Natural Language Processing (NLP) techniques to analyze the sentiment. The analysis results are displayed using interactive visualizations. This is part of the Big Data course being undertaken by me as a student of Swinburne University of Technology.

### **Step 1: Clone the Repository**

Clone this repository to your local machine using the command:

```bash
git clone https://github.com/adnan-techme/yt-sentiment
```
### **Step 2: Activate Virtual Environment**
On Windows:
```bash
python -m venv env
.\env\Scripts\activate
```
On Mac:
```bash
python3 -m venv env
source env/bin/activate
```
### **Step 3: Install Dependencies**
Install all required dependencies:
```bash
pip install -r requirements.txt
```
### **Step 4: Insert API Key**
Add your YouTube API key in the app.py file (refer to the project report for my key)
### **Step 5: Run the Web App**
Use the following command to run the app:
```bash
python app.py
```
The app will be accessible at http://127.0.0.1:5000/. 
