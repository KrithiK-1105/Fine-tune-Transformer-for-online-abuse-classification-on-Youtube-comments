import streamlit as st
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

# Load tokenizer and model (Ensure paths are correct)
model_path = "model"
tokenizer = AlbertTokenizer.from_pretrained(model_path)
model = AlbertForSequenceClassification.from_pretrained(model_path)

# Move model to device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit UI
st.title("Abuse Detection using ALBERT")
st.write("Enter a sentence to check if it's abusive or not.")

# User input
user_text = st.text_area("Enter your text here:")

if st.button("Prediction"):
    if user_text.strip():  # Ensure input is not empty
        # Tokenize input
        tokenized_inputs = tokenizer([user_text], padding="max_length", truncation=True, max_length=512)

        # Convert to PyTorch tensors
        input_ids = torch.tensor(tokenized_inputs["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized_inputs["attention_mask"]).to(device)

        # Make predictions
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # Display result
        if prediction == 1:
            st.error("🔴 This text is classified as **Abusive**.")
        else:
            st.success("🟢 This text is classified as **Non-Abusive**.")
    else:
        st.warning("⚠️ Please enter a sentence before clicking the button.")

# Run with: streamlit run app.py
