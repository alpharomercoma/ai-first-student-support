import os
import openai
import streamlit as st
import base64

# Set up the page configuration
st.set_page_config(
    page_title="🌃 Midnight Mentor",
    page_icon='./images/logo.png',
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to set the background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    background_style = f"""
    <style>
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/jpg;base64,{image_data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        filter: brightness(0.3) blur(5px);
        z-index: -1;
    }}
    .stApp {{
        background: rgba(0, 0, 0, 0.7);
    }}
    .stChatMessage {{
        width: 80%;
        margin: 1rem auto !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    .stChatMessage div[data-testid="stMarkdownContainer"] {{
        color: #ffffff;
        font-size: 1.1rem;
        line-height: 1.5;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.1);
    }}
    .stChatMessage.user-message {{
        background-color: rgba(44, 44, 60, 0.95) !important;
        margin-left: 20% !important;
        border-left: 4px solid #dec960;
    }}
    .stChatMessage.assistant-message {{
        background-color: rgba(66, 66, 99, 0.95) !important;
        margin-right: 20% !important;
        border-left: 4px solid #03a9f4;
    }}
    .stChatInputContainer {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1.5rem;
        background-color: rgba(20, 20, 30, 0.95);
        backdrop-filter: blur(10px);
        border-top: 2px solid rgba(222, 201, 96, 0.2);
        z-index: 1000;
    }}
    [data-testid="stChatMessageContainer"] {{
        padding: 1rem;
        margin-bottom: 80px;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Custom CSS for enhanced styling
def load_custom_css():
    custom_css = """
    <style>
    :root {
        --primary-color: #262730;
        --secondary-color: #03a9f4;
        --background-color: #1b1b1b;
        --text-color: #ffffff;
        --accent-color: #dec960;
    }
    body {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }
    .stSidebar {
        background-color: rgba(20, 20, 30, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    .stSidebar .stMarkdown {
        color: #ffffff;
    }
    .stButton>button {
        background-color: var(--accent-color);
        color: var(--primary-color);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: scale(1.05);
    }
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    .stTextInput input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 2px rgba(222, 201, 96, 0.2);
        background-color: rgba(255, 255, 255, 0.15);
    }
    h1 {
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        letter-spacing: 1px;
    }
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(222, 201, 96, 0.5);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(222, 201, 96, 0.7);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Main application class
class MidnightMentorApp:
    def __init__(self):
        # Initialize session state variables
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'api_key_validated' not in st.session_state:
            st.session_state.api_key_validated = False

    def chat_message(self, content, role='assistant'):
        # Add message to chat history
        st.session_state.chat_history.append({
            'role': role,
            'content': content
        })

        # Display message
        with st.chat_message(role):
            st.markdown(content)

    def get_ai_response(self, user_input):
        # System prompt for AI interaction
        system_prompt = """
        You are Midnight Mentor, an AI assistant designed to help users with various tasks.
        You have multiple capabilities including:
        - Project Idea Generation
        - Python Library Expertise
        - Complex Topic Breakdown
        Provide concise, helpful, and creative responses.
        """

        # Generate response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def run(self):
        # Main content area
        st.markdown("<h1 style='text-align:center; color:#dec960;'>🌃 Midnight Mentor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#dec960;'><b>Made with 💙 by Team Amber!!<b></p>", unsafe_allow_html=True)

        st.markdown(
            "### Available Agents:\n"
            "- **🤖 Orchestrator Agent**: Manage and coordinate multiple AI agents.\n"
            "- **💡 Project Idea Generator Agent**: Ideate creative and practical project concepts.\n"
            "- **🐍 Python Library Expert**: Get expert guidance on Python libraries.\n"
            "- **📖 Breakdown**: Simplify complex topics into digestible information.\n"
        )

        st.markdown(
    "### How It Works:\n"
    "1. **Content Analysis**: Midnight Mentor's AI analyzes the article, identifying core elements like pivotal events, key figures, and essential data points.\n"
    "2. **Information Extraction**: We focus on extracting factual, relevant details—minimizing fluff, opinions, and noise.\n"
    "3. **Structured Insights**: The AI organizes the information into a clear, easy-to-follow summary with actionable points and concise clarity.\n"
    "4. **Objective Presentation**: Stay confident with accurate, unbiased summaries presented in an engaging, no-nonsense style.\n"
    "\n"
    "### Benefits\n"
    "- **Save Time**: Gain key insights in seconds without wading through long-winded articles.\n"
    "- **Boost Comprehension**: Simplify complex ideas for effortless understanding and actionable takeaways.\n"
    "- **Stay Informed**: Keep up-to-date with trends, ideas, and innovations across industries, effortlessly.\n"
    "- **Perfect for Learning**: Enhance your research, studies, or career development with fast, reliable summaries.\n"
    "Midnight Mentor is your **personal knowledge companion**—ensuring you learn smarter, grow faster, and stay ahead of the curve, one summary at a time."
)

        # Sidebar for API key input
        with st.sidebar:
            st.image('./images/logo.png')

            # API key input
            api_key_container = st.empty()
            openai.api_key = api_key_container.text_input(
                'Enter OpenAI API token:',
                type='password',
                placeholder='Your API token here'
            )

            # API key validation
            if not openai.api_key:
                st.warning('Please enter your OpenAI API token!', icon='⚠️')
                return

            try:
                # Validate API key
                openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                st.session_state.api_key_validated = True
                st.success('API Key Validated! Ready to chat.', icon='🎉')
            except openai.error.AuthenticationError:
                st.error('Invalid API key. Please check your token.', icon='🚫')
                return
            except Exception as e:
                st.error(f'An error occurred: {str(e)}', icon='⚠️')
                return

        # Chat container
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])

            # Initial welcome message
            if len(st.session_state.chat_history) == 0:
                self.chat_message("Hello! I'm Midnight Mentor. How can I assist you today? I can help with project ideas, Python libraries, or break down complex topics.")

        # User input
        if st.session_state.api_key_validated:
            user_input = st.chat_input("How can I help you?")

            if user_input:
                # User message
                self.chat_message(user_input, role='user')

                # Get AI response
                ai_response = self.get_ai_response(user_input)

                # AI response
                self.chat_message(ai_response)

# Main app execution
def main():
    # Set background and load custom CSS
    set_background("./images/studio.jpg")
    load_custom_css()

    # Initialize and run the app
    app = MidnightMentorApp()
    app.run()

if __name__ == "__main__":
    main()