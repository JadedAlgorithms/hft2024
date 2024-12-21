# Importing the transformers module
from transformers import pipeline

# Example: Using a pre-trained model for sentiment analysis
classifier = pipeline("sentiment-analysis")

# Test with some text
result = classifier("Hackathons are a great way to learn and innovate!")
print(result)
