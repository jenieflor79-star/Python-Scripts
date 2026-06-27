import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import os

# Set working directory
project_dir = r"C:\Users\4Ps BARMM\Desktop\PythonProject3"
os.chdir(project_dir)

# 1. I-load ang buong dataset
print("Loading the full population dataset...")
df_master = pd.read_csv("PNA.csv")

# 2. I-initialize ang RoBERTa pipeline
print("Initializing RoBERTa transformer model...")
roberta_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

label_mapping = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}

def predict_roberta_sentiment(text):
    try:
        result = roberta_analyzer(str(text)[:512])[0]
        return label_mapping[result['label']]
    except Exception as e:
        return "Error"

# 3. I-deploy ang RoBERTa (Full Population)
print("Deploying RoBERTa model across the whole population dataset...")
df_master['RoBERTa_Full_Sentiment'] = df_master['headlines'].apply(predict_roberta_sentiment)

# 4. I-export ang resulta
output_filename = "BARMM_Full_Population_Sentiment.xlsx"
df_master.to_excel(output_filename, index=False)
print(f"Deployment successful! Saved as: {output_filename}")

# 5. Visualizing: Pie Chart Generation
print("Generating visualization with custom colors...")
sentiment_counts = df_master['RoBERTa_Full_Sentiment'].value_counts()

# Neutral: Blue, Negative: Red, Positive: Dark Green
colors = ['#365F91', '#C0392B', '#1B5E20']

plt.figure(figsize=(9, 7))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.2f%%',
        colors=colors, startangle=140, explode=(0.05, 0, 0), shadow=True,
        textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'}) # White text for contrast

# Title
plt.title('Overall Sentiment Distribution (N=1,711)', fontsize=14, fontweight='bold', color='black')

# Lagyan ng Legend para mas malinaw
plt.legend(labels=sentiment_counts.index, loc="best")

plt.savefig("overall_sentiment_pie.png", dpi=300)
plt.show()