from transformers import BertTokenizer, BertForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import torch
import torch.nn.functional as F

# Load the BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Function to get explanations
def explain_with_white_box(texts):
    model.eval()
    
    # Create a variable to store all the HTML content
    full_html = "<html><body>"

    # Iterate over the texts
    for text in texts:
        # Preprocess the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the explanations with transformers-interpret
        explainer = SequenceClassificationExplainer(model, tokenizer)
        
        # Get the predictions (logits)
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()  # Probabilities for each class
        
        # Predicted class
        predicted_class = torch.argmax(logits, dim=1).item()  # Class with the highest probability
        
        # Create a text block with the prediction and probabilities
        class_names = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        prediction_text = f"Predicted class: {class_names[predicted_class]} (Score: {probabilities[predicted_class]:.4f})"
        probabilities_text = "<br>".join([f"{class_names[i]}: {probabilities[i]:.4f}" for i in range(len(probabilities))])

        # Add a header with the current text
        full_html += f"<h2>Explanation for: {text}</h2>"
        
        # Add the probabilities and predicted class
        full_html += f"<p><strong>Predicted Sentiment:</strong> {prediction_text}</p>"
        full_html += f"<p><strong>Probabilities for each class:</strong></p>"
        full_html += f"<p>{probabilities_text}</p>"

        # Get the HTML visualization as a string
        visualization_html = explainer.visualize().data

        # Concatenate the visualization to the full HTML content
        full_html += str(visualization_html)
        
        # Add a separator between visualizations of different texts
        full_html += "<hr>"

    # Close the <body> and <html> tags
    full_html += "</body></html>"
    
    # Save the entire HTML as a single file
    with open("explanation_visualizations.html", "w", encoding="utf-8") as f:
        f.write(full_html)

    print("Visualizations saved in 'explanation_visualizations.html'.")

# List of texts to generate explanations for
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

# Call the function to generate explanations
explain_with_white_box(texts)
