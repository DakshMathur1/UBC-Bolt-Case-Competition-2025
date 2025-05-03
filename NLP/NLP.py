# Part 1
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the reviews file
file_path = "/Users/daksh/Bolt_Datathon_2025/reviews.txt"
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


import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
from gensim import corpora, models
import spacy

# Load spaCy English model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Load the reviews dataset from your specified file path
file_path = "/Users/daksh/Bolt_Datathon_2025/reviews.txt"
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
        "negative": score["neg"],
        "sentiment": "Positive" if score["compound"] > 0.2 else ("Negative" if score["compound"] < -0.2 else "Neutral")
    })

# Convert to DataFrame
df_reviews = pd.DataFrame(sentiments)

### 1️⃣ **Sentiment Distribution Analysis**
plt.figure(figsize=(8, 4))
sns.countplot(x=df_reviews["sentiment"], palette={"Positive": "green", "Neutral": "gray", "Negative": "red"})
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Reviews")
plt.title("Sentiment Distribution in Guest Reviews")
plt.show()

# Part 2
 
# **Aspect-Based Sentiment Analysis (ABSA)**
# Define categories for safety, pricing, and service
safety_keywords = ["safe", "danger", "accident", "injury", "medical", "rescue", "help", "hospital"]
pricing_keywords = ["expensive", "cheap", "overpriced", "value", "cost", "money", "budget", "price"]
service_keywords = ["staff", "customer", "friendly", "rude", "helpful", "support", "service", "wait"]

# Count occurrences of keywords in each category
aspect_sentiments = {"Safety": 0, "Pricing": 0, "Service": 0}
for review in df_reviews["review"]:
    words = review.lower().split()
    if any(word in words for word in safety_keywords):
        aspect_sentiments["Safety"] += 1
    if any(word in words for word in pricing_keywords):
        aspect_sentiments["Pricing"] += 1
    if any(word in words for word in service_keywords):
        aspect_sentiments["Service"] += 1

# Plot aspect sentiment analysis
plt.figure(figsize=(8, 4))
plt.bar(aspect_sentiments.keys(), aspect_sentiments.values(), color=["blue", "orange", "green"])
plt.xlabel("Aspect")
plt.ylabel("Number of Mentions")
plt.title("Aspect-Based Sentiment Analysis (ABSA)")
plt.show()

### 3️⃣ **TF-IDF Vectorization (Finding Important Words)**
vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
tfidf_matrix = vectorizer.fit_transform(df_reviews["review"])
feature_names = vectorizer.get_feature_names_out()
word_importance = dict(zip(feature_names, tfidf_matrix.sum(axis=0).tolist()[0]))

# Sort words by importance and plot
sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:10]
plt.figure(figsize=(10, 5))
plt.bar(*zip(*sorted_words))
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("TF-IDF Score")
plt.title("Most Important Words in Guest Reviews")
plt.show()

### 4️⃣ **Topic Modeling with LDA (Finding Hidden Themes)**
# Tokenize words and remove stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
tokenized_reviews = [[word for word in review.lower().split() if word.isalpha() and word not in stop_words] for review in df_reviews["review"]]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(tokenized_reviews)
corpus = [dictionary.doc2bow(text) for text in tokenized_reviews]

# Train LDA Model
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

# Print the top words in each topic
for idx, topic in lda_model.show_topics(formatted=True, num_words=10):
    print(f"Topic {idx+1}: {topic}")

### 5️⃣ **Named Entity Recognition (NER)**
entity_counts = Counter()
for review in df_reviews["review"]:
    doc = nlp(review)
    for ent in doc.ents:
        entity_counts[ent.text] += 1

# Get top entities
top_entities = entity_counts.most_common(10)

# Display Named Entities
print("\n🔹 **Most Frequently Mentioned Named Entities**")
for entity, count in top_entities:
    print(f"{entity}: {count} mentions")

# Convert entities to DataFrame for visualization
df_entities = pd.DataFrame(top_entities, columns=["Entity", "Frequency"])

# Plot Named Entity Mentions
plt.figure(figsize=(10, 5))
plt.barh(df_entities["Entity"], df_entities["Frequency"])
plt.xlabel("Frequency")
plt.ylabel("Entities")
plt.title("Top Named Entities in Reviews")
plt.gca().invert_yaxis()
plt.show()



# Part 3

import pandas as pd
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure necessary downloads
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the reviews dataset
file_path = "/Users/daksh/Bolt_Datathon_2025/reviews.txt"
with open(file_path, "r", encoding="utf-8") as file:
    reviews = file.readlines()

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each review
sentiments = []
for review in reviews:
    score = sia.polarity_scores(review)
    sentiments.append({
        "review": review.strip(),
        "compound": score["compound"],
        "sentiment": "Positive" if score["compound"] > 0.2 else ("Negative" if score["compound"] < -0.2 else "Neutral")
    })

# Convert to DataFrame
df_reviews = pd.DataFrame(sentiments)

# Define relevant keywords for price, safety, and negative contexts
pricing_keywords = [
    "price", "expensive", "cheap", "overpriced", "cost", "money", "value", "budget", "refund", "extra",
    "affordability", "hidden", "charges", "fees", "discount", "waste", "worth", "expensive", "ripoff",
    "fair", "billing", "scam", "overcharge", "unreasonable", "unaffordable", "pricing", "expensiveness",
    "underpriced", "payment", "deposit", "surge", "increase", "decrease", "service charge", "inflation",
    "additional", "excess", "undervalued", "financial", "debit", "credit", "transaction", "loan", "wallet",
    "currency", "exchange", "receipt", "prepaid", "subscription", "cash", "installment", "reimbursement"
]

safety_keywords = [
    "safe", "danger", "accident", "injury", "medical", "rescue", "help", "hospital", "sprain", "broken",
    "risk", "patrol", "lifeguard", "first aid", "snow", "avalanche", "fall", "collision", "poor lighting",
    "emergency", "fire", "ambulance", "hazard", "slippery", "unsecure", "unsafe", "fracture", "bleeding",
    "medical staff", "security", "health risk", "paramedic", "disaster", "flood", "earthquake", "tornado",
    "storm", "crash", "evacuation", "death", "serious", "trauma", "heart attack", "infection", "allergy",
    "breathing", "intensive care", "bandage", "burn", "electrical", "shock", "unconscious", "blood loss"
]

# Convert lists to sets
pricing_keywords_set = set(pricing_keywords)
safety_keywords_set = set(safety_keywords)
all_relevant_keywords = pricing_keywords_set.union(safety_keywords_set)  

# Define words to remove (stopwords + common useless words)
common_stopwords = set(stopwords.words("english"))
extra_stopwords = set([
    "is", "was", "the", "a", "if", "this", "that", "and", "to", "it", "on", "of", "for", "in", "with",
    "we", "but", "so", "not", "they", "their", "at", "by", "as", "or", "our", "your", "can", "be",
    "will", "has", "have", "you", "i", "us", "all", "some", "more", "hotel", "room", "ski", "trip",
    "good", "bad", "experience", "thing", "better", "worse", "nice", "poor", "great", "terrible"
])
all_stopwords = common_stopwords.union(extra_stopwords)

# Function to extract only relevant words from reviews
def get_filtered_words(reviews, relevant_keywords, top_n=50):  # Now extracting 50 words
    words = " ".join(reviews).lower().split()
    words = [word for word in words if word in relevant_keywords]  # Keep only relevant words
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

# Extract reviews for each category
negative_reviews = df_reviews[df_reviews["sentiment"] == "Negative"]["review"]
pricing_reviews = df_reviews[df_reviews["review"].str.contains("|".join(pricing_keywords), case=False, na=False)]["review"]
safety_reviews = df_reviews[df_reviews["review"].str.contains("|".join(safety_keywords), case=False, na=False)]["review"]

# Get filtered word frequencies
neg_word_freq = get_filtered_words(negative_reviews, all_relevant_keywords, top_n=50)  
price_word_freq = get_filtered_words(pricing_reviews, pricing_keywords_set, top_n=50)
safety_word_freq = get_filtered_words(safety_reviews, safety_keywords_set, top_n=50)

# Convert to DataFrame for visualization
df_neg_words = pd.DataFrame(neg_word_freq, columns=["Word", "Frequency"])
df_price_words = pd.DataFrame(price_word_freq, columns=["Word", "Frequency"])
df_safety_words = pd.DataFrame(safety_word_freq, columns=["Word", "Frequency"])

# Function to plot frequent words
def plot_word_frequencies(df, title, color):
    plt.figure(figsize=(18, 7))
    plt.bar(df["Word"], df["Frequency"], color=color)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

# Plot results
plot_word_frequencies(df_neg_words, "Top 50 Frequent Words in Negative Reviews", "red")
plot_word_frequencies(df_price_words, "Top 50 Frequent Words in Pricing-Related Reviews", "blue")
plot_word_frequencies(df_safety_words, "Top 50 Frequent Words in Safety-Related Reviews", "green")

# Display DataFrames
import ace_tools as tools
tools.display_dataframe_to_user(name="Negative Reviews Word Frequency", dataframe=df_neg_words)
tools.display_dataframe_to_user(name="Pricing Reviews Word Frequency", dataframe=df_price_words)
tools.display_dataframe_to_user(name="Safety Reviews Word Frequency", dataframe=df_safety_words)

