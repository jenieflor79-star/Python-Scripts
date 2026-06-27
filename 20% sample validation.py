import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.model_selection import train_test_split

# Download the VADER lexicon dependencies safely
nltk.download('vader_lexicon', quiet=True)

print("🚀 Step 1: Loading and preprocessing dataset...")
# Load dataset
df_full = pd.read_csv("PNA.csv")

# Create a clean, reproducible index ID column starting from 1
df_full['id'] = df_full.index + 1

# Isolate the exact same 20% random sample used for human annotation
_, df_validation = train_test_split(df_full, test_size=0.20, random_state=42)
df_validation = df_validation.copy()

print(f"✅ Successfully isolated {len(df_validation)} validation headlines.")

# -------------------------------------------------------------------------
# TRACK B: VADER Sentiment Analysis
# -------------------------------------------------------------------------
print("\n🤖 Track B: Running VADER Lexicon Model...")
analyzer = SentimentIntensityAnalyzer()


def get_vader_label(text):
    # Get numeric polarization scores
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']

    # Apply standard academic threshold cutoffs
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


# Apply VADER to the 342 headlines
df_validation['vader_prediction'] = df_validation['headlines'].apply(get_vader_label)
print("✅ VADER processing complete.")

# -------------------------------------------------------------------------
# TRACK C: RoBERTa Zero-Shot Classification
# -------------------------------------------------------------------------
print("\n🤗 Track C: Loading RoBERTa Zero-Shot Classification Pipeline...")
# This automatically loads a highly optimized, out-of-the-box NLI model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",  # Standard transformer baseline for zero-shot text mining
    device=-1  # Set to 0 if you have an active Nvidia GPU configured
)

candidate_labels = ["positive", "neutral", "negative"]
roberta_preds = []

print("⏳ Running RoBERTa inference over 342 headlines (This may take a minute)...")
for idx, row in enumerate(df_validation['headlines'], 1):
    # Run the zero-shot inference pipeline
    result = classifier(str(row), candidate_labels=candidate_labels)
    # The highest probability label is assigned to top slot index 0
    top_label = result['labels'][0]
    roberta_preds.append(top_label)

    if idx % 50 == 0:
        print(f"   Processed {idx}/342 headlines...")

df_validation['roberta_prediction'] = roberta_preds
print("✅ RoBERTa transformer processing complete.")

# -------------------------------------------------------------------------
# Save Combined Automated Predictions
# -------------------------------------------------------------------------
output_filename = "RoBERTA VS VADER VS Manual.xlsx"
df_validation[['id', 'headlines', 'vader_prediction', 'roberta_prediction']].to_excel(output_filename, index=False)

print(f"\n🎉 SUCCESS: Automated outputs saved to '{output_filename}'")
print("You can now match these row IDs against your human expert's labels for evaluation!")