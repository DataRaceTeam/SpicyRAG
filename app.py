import streamlit as st
import requests

# Set up the Streamlit page configuration
st.set_page_config(page_title="NPA knowledge Ecosystem", page_icon="_")

# App title
st.title("NPA knowledge Ecosystem")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to send message to FastAPI backend
def send_message(user_input):
    # Define the API endpoint
    api_url = "http://interface:8000/ask/"

    # Prepare the payload
    payload = {"text": user_input}

    # Send POST request to FastAPI backend
    response = requests.post(api_url, json=payload)

    # Check if request was successful
    if response.status_code == 200:
        return response.json()["llm_response"]
    else:
        return "Error: Could not reach the backend."


# User input text box
user_input = st.text_area("Case to search definition in NPA knowledge:", height=150)

# If user submits a message
if st.button("Send") and user_input:
    # Display the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send message to backend and get the response
    response_content = send_message(user_input)

    # Display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Display the chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Assistant: {message['content']}")
