from flask import Flask, request, render_template
import pandas as pd
import sklearn
import flask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# Initialize Flask app
app = Flask(__name__)


# Step 2: Prepare Sample Dataset
# Sample dataset for demonstration
data = {
    'url': [
        'https://www.google.com', 
        'http://malicious.com/badsite', 
        'https://facebook.com/login', 
        'http://phishing.com/login.php',
        'https://github.com', 
        'http://badwebsite.com/malware',
        'https://linkedin.com/in/login', 
        'http://fakebank.com/login',
        'https://twitter.com', 
        'http://harmfulsite.org/download'
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = safe, 1 = unsafe
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Extract Features from URLs using TF-IDF
vectorizer = TfidfVectorizer(token_pattern=r'[A-Za-z0-9]+', max_features=3000)
X = vectorizer.fit_transform(df['url'])  # Feature matrix
y = df['label']  # Labels

# Step 4: Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Route to display the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the URL entered by the user
    url = request.form['url']

    # Transform the URL to match the training data features
    url_features = vectorizer.transform([url])

    # Predict if the URL is safe or unsafe
    prediction = model.predict(url_features)
    result = "unsafe" if prediction[0] == 1 else "safe"

    # Render the result on the HTML page
    return render_template('index.html', prediction_text=f"The URL '{url}' is {result}.")

# Run the Flask app with environment-specific host and port
import os

if __name__ == "__main__":
    # Use the PORT environment variable, defaulting to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

