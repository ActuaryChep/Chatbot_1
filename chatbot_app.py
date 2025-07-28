#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit openai scikit-learn pandas


# In[2]:


import streamlit as st
import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------- CONFIGURATION ---------------------
openai.api_key = "YOUR_OPENAI_API_KEY"  # 🔁 Replace with your API key

# Load the dialogs.txt file
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["user", "bot"])
    df.dropna(inplace=True)
    return df

data = load_data("C:/Users/Me/Downloads/chatbotconversation/dialogs.txt")

# Vectorize the user questions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['user'])

# --------------------- RESPONSE FUNCTIONS ---------------------

def get_faq_response(query, threshold=0.6):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    max_sim = similarity.max()

    if max_sim >= threshold:
        best_idx = similarity.argmax()
        return data.iloc[best_idx]['bot']
    else:
        return None

def ask_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------- STREAMLIT UI ---------------------

st.set_page_config(page_title="Smart Customer Support Chatbot", page_icon="🤖")
st.title("🤖 Smart Customer Support Chatbot")
st.markdown("Ask me a question. I'll answer from known support dialogues or use AI if needed.")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat input
user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.chat.append(("user", user_input))

    # Try to match from FAQ
    response = get_faq_response(user_input)
    if response:
        bot_reply = f"🧠 (From FAQ): {response}"
    else:
        gpt_reply = ask_gpt(user_input)
        bot_reply = f"🤖 (From GPT): {gpt_reply}"

    st.session_state.chat.append(("bot", bot_reply))

# Display chat history
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

if st.button("🔄 Clear Chat"):
    st.session_state.chat = []


# In[ ]:




