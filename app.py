import streamlit as st
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "./t5_sentiment_response_model_final"

st.set_page_config(
    page_title="Polite Response Generator",
    page_icon="💬",
    layout="centered"
)

# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# --------------------------------------------------
# Generate Response
# --------------------------------------------------

def generate_response(text):

    cleaned_text = clean_text(text)

    prompt = f"Generate a polite response: {cleaned_text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    # Make sure inputs and model are on the same device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    response = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return cleaned_text, response


# --------------------------------------------------
# Streamlit Interface
# --------------------------------------------------

st.title("💬 Polite Response Generator")

st.write(
    "Generate a polite response to a message using a fine-tuned T5-small model."
)

st.divider()

user_text = st.text_area(
    "Enter a message",
    placeholder="Example: I am really disappointed with the service.",
    height=150
)

generate_button = st.button(
    "✨ Generate Polite Response",
    type="primary"
)

if generate_button:

    if not user_text.strip():

        st.warning("Please enter a message first.")

    else:

        with st.spinner("Generating response..."):

            cleaned_text, response = generate_response(
                user_text
            )

        st.subheader("🧹 Cleaned Text")

        st.info(cleaned_text)

        st.subheader("🤖 Generated Response")

        st.success(response)


# --------------------------------------------------
# Model Information
# --------------------------------------------------

with st.expander("About this model"):

    st.write(
        """
        **Model:** T5-small

        **Training:** Fine-tuned using Hugging Face Transformers

        **Dataset:** TweetEval sentiment dataset

        **Task:** Generate polite responses based on sentiment.

        **Processing:**
        - URL removal
        - Mention removal
        - Hashtag removal
        - Special-character removal
        - Text normalization
        """
    )

    st.write(
        f"Running on: `{device}`"
    )
