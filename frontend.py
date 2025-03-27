import streamlit as st
import requests
import pandas as pd
import time
import plotly.express as px
from datetime import datetime
import json
import os

# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL")

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_user_id" not in st.session_state:
    # Generate a simple random user ID
    import uuid
    st.session_state.current_user_id = str(uuid.uuid4())[:8]
if "prompt_id" not in st.session_state:
    st.session_state.prompt_id = None
if "conversation_ids" not in st.session_state:
    st.session_state.conversation_ids = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()
if "prompts_data" not in st.session_state:
    st.session_state.prompts_data = []

# Function to get available prompts
def get_prompts():
    try:
        response = requests.get(f"{API_BASE_URL}/prompts")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch prompts: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

# Function to send a message to the chatbot
def send_message(user_message, prompt_id=None):
    try:
        payload = {
            "user_id": st.session_state.current_user_id,
            "message": user_message,
        }
        if prompt_id:
            payload["prompt_id"] = prompt_id
            
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            # Store the conversation id for feedback later
            st.session_state.prompt_id = data["prompt_id"]
            # We'll assume the conversation_id is returned or can be derived
            # In a real implementation, you'd store the actual conversation ID
            conversation_id = len(st.session_state.conversation_ids) + 1
            st.session_state.conversation_ids.append(conversation_id)
            return data["response"], conversation_id
        else:
            st.error(f"Failed to get response: {response.text}")
            return "Sorry, I couldn't process your request.", None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return "Sorry, I'm having trouble connecting to the server.", None

# Function to submit feedback
def submit_feedback(conversation_id, score, feedback_text=""):
    try:
        payload = {
            "conversation_id": conversation_id,
            "feedback_score": score,
            "feedback_text": feedback_text
        }
        response = requests.post(f"{API_BASE_URL}/feedback", json=payload)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to submit feedback: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return False

# Sidebar - control panel
with st.sidebar:
    st.title("🤖 Chatbot Controls")
    st.divider()
    
    # User ID display
    st.subheader("User Information")
    st.info(f"User ID: {st.session_state.current_user_id}")
    
    # Prompt selection
    st.subheader("Available Prompts")
    
    if st.button("Refresh Prompts"):
        with st.spinner("Fetching prompts..."):
            st.session_state.prompts_data = get_prompts()
    
    # Initial load of prompts
    if not st.session_state.prompts_data:
        with st.spinner("Fetching prompts..."):
            st.session_state.prompts_data = get_prompts()
    
    # Create a DataFrame for display
    if st.session_state.prompts_data:
        prompts_df = pd.DataFrame(st.session_state.prompts_data)
        selected_prompt = st.selectbox(
            "Select Prompt Template",
            options=prompts_df["id"],
            format_func=lambda x: f"Prompt {x} (v{prompts_df.loc[prompts_df['id'] == x, 'version'].values[0]})"
        )
        
        if selected_prompt:
            st.session_state.prompt_id = selected_prompt
            prompt_content = prompts_df.loc[prompts_df["id"] == selected_prompt, "content"].values[0]
            with st.expander("View Prompt Content"):
                st.text_area("Prompt", value=prompt_content, height=200, disabled=True)
    else:
        st.warning("No prompt templates available")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    if st.session_state.prompts_data:
        # Create a simple bar chart of prompt performance
        fig = px.bar(
            prompts_df,
            x="id",
            y="score",
            color="version",
            labels={"id": "Prompt ID", "score": "Performance Score", "version": "Version"},
            title="Prompt Performance"
        )
        st.plotly_chart(fig, use_container_width=True)

# Main chat interface
st.title("🤖 AI Chatbot Optimization")
st.caption("Chat with the AI and help improve its responses through feedback")

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Only show feedback options for assistant messages if feedback hasn't been given yet
        if message["role"] == "assistant" and i not in st.session_state.feedback_given:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 5])
            with col1:
                if st.button("😞", key=f"bad_{i}"):
                    if submit_feedback(message["conversation_id"], 0.0):
                        st.session_state.feedback_given.add(i)
                        st.rerun()
            with col2:
                if st.button("😐", key=f"neutral_{i}"):
                    if submit_feedback(message["conversation_id"], 0.5):
                        st.session_state.feedback_given.add(i)
                        st.rerun()
            with col3:
                if st.button("😊", key=f"good_{i}"):
                    if submit_feedback(message["conversation_id"], 1.0):
                        st.session_state.feedback_given.add(i)
                        st.rerun()
            with col4:
                if st.button("Custom", key=f"custom_{i}"):
                    message["show_custom"] = True
                    st.rerun()
            
            # Show custom feedback form if requested
            if message.get("show_custom", False):
                with st.form(key=f"custom_feedback_{i}"):
                    feedback_score = st.slider("Rate this response:", 0.0, 1.0, 0.5, step=0.1)
                    feedback_text = st.text_area("Additional comments:")
                    submit_button = st.form_submit_button("Submit Feedback")
                    
                    if submit_button:
                        if submit_feedback(message["conversation_id"], feedback_score, feedback_text):
                            st.session_state.feedback_given.add(i)
                            st.rerun()

# User input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show the message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response from API
    with st.spinner("Thinking..."):
        response_text, conversation_id = send_message(prompt, st.session_state.prompt_id)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "conversation_id": conversation_id
    })
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response_text)

# Add educational information at the bottom
with st.expander("About Prompt Engineering"):
    st.write("""
    ## What is Prompt Engineering?
    
    Prompt engineering is the process of designing and optimizing prompts to effectively communicate with AI systems like this one.
    A well-engineered prompt helps the AI understand exactly what you need and generates better responses.
    
    ## How This App Works
    
    1. You chat with the AI using different prompt templates
    2. You provide feedback on the AI's responses
    3. The system uses your feedback to optimize the prompts
    4. Over time, the AI's responses should improve based on this feedback loop
    
    ## Tips for Good Prompts
    
    - Be specific and clear about what you need
    - Provide context when needed
    - Structure complex requests as step-by-step instructions
    - Include examples if appropriate
    """)

# Footer
st.divider()
st.caption("AI-Powered Chatbot Optimization © 2025")