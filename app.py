import os
import openai
import streamlit as st
import base64
from swarm import Agent, Swarm
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
import numpy as np
import faiss
from duckduckgo_search import DDGS
import pandas as pd

# Configuration and Environment Setup
class MidnightMentorConfig:
    def __init__(self):
        # Ensure environment variables are set
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OpenAI API Key not set in environment variables.")

# PDF Processing Utility
class PDFSwarmExtractor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

    def process_single_pdf(self, pdf_path: str) -> List[str]:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = self.text_splitter.split_documents(pages)
            texts = [chunk.page_content for chunk in chunks]
            print(f"Successfully processed {pdf_path}")
            return texts
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []

    def process_pdf_directory(self, directory_path: str) -> dict:
        pdf_files = [str(f) for f in Path(directory_path).glob("**/*.pdf")]
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pdf = {executor.submit(self.process_single_pdf, pdf): pdf for pdf in pdf_files}

            for future in future_to_pdf:
                pdf_path = future_to_pdf[future]
                try:
                    texts = future.result()
                    results[pdf_path] = texts
                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")
                    results[pdf_path] = []

        return results

# Utility Functions for Embeddings and Search
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def create_embeddings_and_retrieve(query, texts):
    embeddings = [get_embedding(doc) for doc in texts]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    query_embedding = get_embedding(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [texts[i] for i in indices[0]]
    return ' '.join(retrieved_docs)

def generate_project_ideas(context):
    prompt = f"""
    Based on the following context, generate innovative project ideas:
    {context}

    Please provide a list of project ideas that are relevant and actionable.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a project idea generator."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Midnight Mentor Streamlit Application
class MidnightMentorApp:
    def __init__(self):
        # Initialize session state variables
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'api_key_validated' not in st.session_state:
            st.session_state.api_key_validated = False

        # Initialize SWARM components
        self.initialize_swarm_agents()

    def initialize_swarm_agents(self):
        # Initialize Swarm client
        self.swarm_client = Swarm()

        # Project Idea Generator Agent
        self.project_generator_agent = Agent(
            name="Project Idea Generator Agent",
            model="gpt-4o-mini",
            instructions="""
            Generate project ideas based on lessons learned from an AI Engineering Bootcamp.
            Consider various aspects such as machine learning, data analysis, and AI ethics.
            """,
            functions=[
                create_embeddings_and_retrieve,
                generate_project_ideas
            ]
        )

        # Web Search Agent
        self.web_search_agent = Agent(
            name="Web Search Agent",
            instructions="You are a website search agent specialized in searching website content.",
            functions=[self.web_search]
        )

        # Python Expert Agent
        self.python_expert_agent = Agent(
            name="Python Expert Agent",
            model="gpt-4o-mini",
            instructions="""
            You are a Python Expert AI assistant. Your task is to suggest Python libraries
            that are highly suitable for the given query. Provide a brief explanation for
            each library you recommend, focusing on its relevance and key features.
            """
        )

        # Student Support (Orchestrator) Agent
        self.student_support_agent = Agent(
            name="Student Support Agent",
            instructions="""
            You are a student support agent that accepts bootcamp student's requests
            and calls a tool to transfer to the right intent.
            Transfer to the appropriate agent based on the request.
            """
        )

    def web_search(self, query):
        results = DDGS().text(
            keywords=query,
            region='wt-wt',
            safesearch='off',
            timelimit='7d',
            max_results=10
        )
        return pd.DataFrame(results)

    def chat_message(self, content, role='assistant'):
        st.session_state.chat_history.append({
            'role': role,
            'content': content
        })
        with st.chat_message(role):
            st.markdown(content)

    def get_ai_response(self, user_input):
        try:
            # Determine the most appropriate agent based on input
            if "project" in user_input.lower():
                response = self.swarm_client.run(
                    agent=self.project_generator_agent,
                    messages=[{"role": "user", "content": user_input}]
                )
            elif "python" in user_input.lower():
                response = self.swarm_client.run(
                    agent=self.python_expert_agent,
                    messages=[{"role": "user", "content": user_input}]
                )
            else:
                # Default to student support agent for routing
                response = self.swarm_client.run(
                    agent=self.student_support_agent,
                    messages=[{"role": "user", "content": user_input}]
                )

            return response.messages[-1]["content"]
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def run(self):
        # Streamlit UI setup (similar to previous implementation)
        st.markdown("<h1 style='text-align:center; color:#dec960;'>🌃 Midnight Mentor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#dec960;'><b>Made with 💙 by Team Amber!!</b></p>", unsafe_allow_html=True)

        st.markdown(
            "### Available Agents:\n"
            "- **🤖 Orchestrator Agent**: Manage and coordinate multiple AI agents\n"
            "- **💡 Project Idea Generator Agent**: Ideate creative project concepts\n"
            "- **🐍 Python Library Expert**: Get expert guidance on Python libraries\n"
            "- **📖 Breakdown Agent**: Simplify complex topics\n"
        )

        # Sidebar for API key input
        with st.sidebar:
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

# Main application execution
def main():
    # Configuration setup
    config = MidnightMentorConfig()

    # Initialize and run the app
    app = MidnightMentorApp()
    app.run()

if __name__ == "__main__":
    main()