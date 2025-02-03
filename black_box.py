import shap
import transformers

# Load the sentiment analysis pipeline
classifier = transformers.pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", return_all_scores=True)

# List of texts for explanation generation
texts = [
    "I love spending time with my friends.",
    "Today has been a wonderful day.",
    "The service at this restaurant was exceptional.",
    "I can't believe I failed my exam.",
    "The weather is horrible, everything is wet and cold.",
    "The food was completely tasteless, I won't try it again.",
    "Today was a terrible day, nothing went right.",
    "The day was interesting, but I can't tell if it was good or bad.",
    "The movie was entertaining, but it didn't make me feel anything particular.",
    "I feel a bit tired, but I still want to keep working.",
    "The event was nice, but some aspects could have been improved."
]

# Create the SHAP explainer
explainer = shap.Explainer(classifier)

# Compute SHAP values for the sample sentences
shap_values = explainer(texts)

# Generate the visualization as an HTML file
html_explanation = shap.plots.text(shap_values, display=False)

# Save the visualization to an HTML file
with open("shap_explanation.html", "w", encoding="utf-8") as f:
    f.write(html_explanation)

print("Explanation saved in 'shap_explanation.html'. Open it in a browser to view the results.")






