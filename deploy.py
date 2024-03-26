import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import streamlit as st

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('chat_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('chat_model')

# Load the classes
with open('chat_model/classes.json', 'r') as file:
    classes = json.load(file)

# Function to get the prediction
def get_prediction(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=50, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits
    prediction = torch.argmax(logits, dim=1).item()

    return classes[prediction]

# Streamlit app
def main():
    st.title("SAIL Chatbot")
    st.write("Welcome to the SAIL Chatbot! Ask me anything related to your college's student portal.")

    user_input = st.text_input("You: ", "")
    if user_input:
        prediction = get_prediction(user_input)
        st.write(f"SAIL: {prediction}")

if __name__ == "__main__":
    main()
