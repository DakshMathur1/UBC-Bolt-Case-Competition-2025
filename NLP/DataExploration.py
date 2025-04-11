import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the reviews file
file_path = "reviews.txt"
with open(file_path, "r", encoding="utf-8") as file:
    reviews = file.readlines()

# Initialize Sentiment Intensity Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each review
sentiments = []
for review in reviews:
    score = sia.polarity_scores(review)
    sentiments.append({
        "review": review.strip(),
        "compound": score["compound"],
        "positive": score["pos"],
        "neutral": score["neu"],
        "negative": score["neg"]
    })

# Convert to DataFrame
df_reviews = pd.DataFrame(sentiments)

# Classify sentiment categories
df_reviews["sentiment"] = df_reviews["compound"].apply(lambda x: "Positive" if x > 0.2 else ("Negative" if x < -0.2 else "Neutral"))

# Count sentiment occurrences
sentiment_counts = df_reviews["sentiment"].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(8, 4))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "gray", "red"])
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Sentiment Distribution in Guest Reviews")
plt.show()

# Generate Word Cloud for most common words
all_reviews = " ".join(df_reviews["review"]).lower()
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Reviews")
plt.show()

# Display first few rows of sentiment analysis results
print(df_reviews.head())
