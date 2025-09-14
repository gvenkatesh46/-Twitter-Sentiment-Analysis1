import streamlit as st
import tweepy
import pickle
import subprocess
import sys
import re

# ================= Replace with your Bearer Token =================
BEARER_TOKEN = "A"
# ==================================================================

# Initialize Twitter API client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Load sentiment model and vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()


# ---------------- Sentiment Prediction ----------------
def predict_sentiment(text):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 2)
    else:
        confidence = None

    sentiment = "Positive 😀" if prediction == 1 else "Negative 😡"
    return sentiment, confidence


# ---------------- Fetch Tweets (API) ----------------
def get_tweets_api(username, count=5):
    try:
        user = client.get_user(username=username)
        tweets = client.get_users_tweets(
            id=user.data.id,
            max_results=min(count, 100),
            tweet_fields=["created_at", "text"]
        )
        if tweets.data:
            return [t.text for t in tweets.data]
        else:
            return []
    except tweepy.TooManyRequests:
        st.warning("⚠️ Twitter API rate limit hit. Switching to snscrape...")
        return None
    except Exception as e:
        st.error(f"Twitter API error: {e}")
        return None


# ---------------- Fetch Tweets (snscrape fallback) ----------------
def get_tweets_snscrape(username, count=5):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "snscrape", "--max-results", str(count), f"twitter-user:{username}"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        tweets = []
        for line in lines:
            match = re.search(r"content='(.*?)'", line)
            if match:
                tweets.append(match.group(1))
        return tweets[:count]
    except Exception as e:
        st.error(f"snscrape error: {e}")
        return []


# ---------------- Streamlit UI ----------------
st.title("🐦 Twitter Sentiment Analysis")

option = st.selectbox("Choose an option", ["Input text", "Fetch tweets from user"])

# ---- Input Text ----
if option == "Input text":
    text_input = st.text_area("Enter text to analyze sentiment")
    if st.button("Analyze"):
        if text_input.strip():
            sentiment, confidence = predict_sentiment(text_input)
            conf_text = f" ({confidence}% confidence)" if confidence else ""
            st.markdown(f"""
            <div style="background-color:#000000; padding:10px; border-radius:8px; margin:10px 0; color:white;">
                <p><b>Text:</b> {text_input}</p>
                <p><b>Sentiment:</b> {sentiment}{conf_text}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text!")

# ---- Fetch Tweets ----
elif option == "Fetch tweets from user":
    username = st.text_input("Enter Twitter username (without @)", value="nasa")
    if st.button("Fetch & Analyze Tweets"):
        if username.strip():
            # Try Twitter API first
            tweets = get_tweets_api(username, count=5)

            # Fallback to snscrape if API failed
            if tweets is None or len(tweets) == 0:
                tweets = get_tweets_snscrape(username, count=5)

            if tweets:
                for tweet in tweets:
                    sentiment, confidence = predict_sentiment(tweet)
                    conf_text = f" ({confidence}% confidence)" if confidence else ""
                    st.markdown(f"""
                    <div style="background-color:#000000; padding:10px; border-radius:8px; margin:10px 0; color:white;">
                        <p><b>Tweet:</b> {tweet}</p>
                        <p><b>Sentiment:</b> {sentiment}{conf_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No tweets found or unable to fetch tweets.")
        else:
            st.warning("Please enter a username.")

