import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import requests
import streamlit as st
import pandas as pd
from pinecone import Pinecone

# --- LangChain Imports ---
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Page Configuration & CSS ---
st.set_page_config(
    page_title="CineMate AI Recommender",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    .st-emotion-cache-1y4p8pa { max-width: 900px; }
    
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(45deg, #1E40AF, #58C4E8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    h5 { color: #E0F2F1; text-align: center; font-weight: 300; margin-bottom: 2rem; }
    
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #58C4E8;
        background-color: transparent;
        color: #58C4E8;
        transition: all 0.3s ease-in-out;
        padding: 0.5rem 1rem;
    }

    .stButton>button:hover {
        background-color: #58C4E8;
        color: #0D2137;
        border-color: #58C4E8;
        box-shadow: 0 4px 20px 0 rgba(88, 196, 232, 0.4);
    }
    
    .stForm {
        background-image: linear-gradient(to bottom right, rgba(22, 58, 92, 0.7), rgba(13, 33, 55, 0.7));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem 3rem; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-top: 2rem;
        transition: all 0.3s ease-in-out; 
    }

    .stForm:hover {
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
        transform: translateY(-5px);
    }

    .stTextInput>div>div>input {
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #E0F2F1;
        font-weight: 300;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #58C4E8;
        box-shadow: 0 0 0 0.2rem rgba(88, 196, 232, 0.5);
    }
    
    [data-testid="stFormSubmitButton"] {
        display: flex;
        justify-content: center; 
        padding-top: 1rem; 
    }

    [data-testid="stFormSubmitButton"] button {
        width: 60%; 
        font-weight: 700;
        background: linear-gradient(45deg, #00a9ff, #58C4E8); 
        color: white;
        border: none;
        box-shadow: 0 4px 15px 0 rgba(0, 169, 255, 0.3);
        transition: all 0.3s ease-in-out;
    }
    
    [data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 6px 25px 0 rgba(0, 169, 255, 0.5);
        transform: translateY(-3px);
    }

    [data-testid="stNotification"][data-st-notification-status="info"] {
        background: rgba(88, 196, 232, 0.2);
        border: 1px solid rgba(88, 196, 232, 0.5);
        border-radius: 15px;
        color: #E0F2F1;
    }

    [data-testid="stNotification"][data-st-notification-status="success"] {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: #E0F2F1;
    }
</style>
""", unsafe_allow_html=True)


class OpenRouterEmbeddings(Embeddings):
    """Custom LangChain Embeddings wrapper for OpenRouter."""
    def __init__(self, api_key: str, model: str = "perplexity/pplx-embed-v1-0.6b"):
        self.api_key = api_key
        self.model = model
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Process multiple documents (used if LangChain tries to index new data)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Process a single user query
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model, 
                "input": text
            }
        )
        if response.status_code != 200:
            raise ValueError(f"OpenRouter Error: {response.text}")
        return response.json()["data"][0]["embedding"]

# --- Core Logic & Caching ---
@st.cache_resource(show_spinner="Initializing AI Services...")
def initialize_services() -> Tuple[Pinecone, PineconeVectorStore, ChatOpenAI, pd.DataFrame]:
    """Idempotent initialization of API clients, Langchain wrappers, and DataFrames."""
    
    # 1. Secret Validation
    pinecone_key = st.secrets.get("PINECONE_API_KEY")
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
    index_name = st.secrets.get("PINECONE_INDEX_NAME", "movie-recommender")

    if not all([pinecone_key, openrouter_key]):
        st.error("Critical Error: Missing API keys in Streamlit Secrets.")
        st.stop()
        
    os.environ["PINECONE_API_KEY"] = pinecone_key

    try:
        # 2. Raw Pinecone Client (Required for Reranking Inference API)
        pc = Pinecone(api_key=pinecone_key)

        # 3. Langchain Vector Store (Bi-Encoder Retrieval)
        embeddings = OpenRouterEmbeddings(api_key=openrouter_key)
        
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        # 4. Langchain LLM (Generative)
        llm = ChatOpenAI(
            model_name="deepseek/deepseek-v4-flash",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=500
        )

        # 5. Metadata Fallback Data
        df = pd.read_csv('MovieData.csv')
        
        return pc, vectorstore, llm, df

    except Exception as e:
        logger.error(f"Service initialization failed: {e}", exc_info=True)
        st.error(f"Failed to connect to backend services: {e}")
        st.stop()


# --- Pipeline Execution Functions ---
def retrieve_and_rerank(query: str, vectorstore: PineconeVectorStore, pc: Pinecone, top_k: int = 5) -> List[Dict[str, Any]]:
    """Executes a broad dense fetch followed by a precision Cross-Encoder rerank."""
    
    # Step 1: Broad Bi-Encoder Fetch (Get top 50 candidates)
    initial_docs = vectorstore.similarity_search(query, k=50)
    
    if not initial_docs:
        return []

    # Step 2: Format payload for Pinecone Inference Reranker
    documents_payload = [
        {"id": str(i), "text": doc.page_content, "metadata": doc.metadata} 
        for i, doc in enumerate(initial_docs)
    ]
    
    # Step 3: Execute Cross-Encoder Reranking
    reranked_results = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents_payload,
        top_n=top_k,
        return_documents=True
    )
    
    # Step 4: Parse final ranked payload
    final_docs = []
    for match in reranked_results.data:
        final_docs.append({
            "text": match.document.text,
            "metadata": match.document.metadata,
            "score": match.score
        })
        
    return final_docs


def generate_llm_recommendation(query: str, ranked_docs: List[Dict[str, Any]], llm: ChatOpenAI) -> str:
    """Passes the precision-ranked documents into the LLM context window."""
    
    template = """
    You are 'CineMate', a friendly and enthusiastic movie expert chatbot. Your goal is to give excellent, personalized movie recommendations by using the provided movie data to answer the user's request.

    **Your Instructions:**
    1.  Analyze the user's request to understand their taste (genres, actors, directors, themes, etc.).
    2.  Use the provided **Context from our movie database** to find a movie that is a great match.
    3.  If you cannot find a good match in the context, say so and suggest that you can search more broadly if they'd like.
    4.  **Crucially, DO NOT spoil major plot twists or endings.**
    5.  Present your recommendation in the exact structured format below.

    **Context from our movie database:**
    {context}

    **User's Request:**
    {question}

    **Your Recommendation:**
    
    **ID**: [Movie ID]
    
    **Movie:** [Movie Title] ([Year])

    **Logline:** [A compelling, one-sentence hook to grab their attention, extracted from the context.]

    **Synopsis:** [A brief, 2-3 sentence summary of the plot without giving too much away, extracted from the context.]

    **Why You'll Like It:** [Directly connect the movie to the user's stated preferences from their request.]

    **Details:**
    * **Genre:** [Primary Genre(s)]
    * **Director:** [Director's Name]
    * **Starring:** [Lead Actor(s)]

    [Ask a follow-up question to keep the conversation going.]
    """
    
    # Compile the highly-relevant context string
    context_blocks = []
    for doc in ranked_docs:
        title = doc["metadata"].get("title", "Unknown Title")
        context_blocks.append(f"Title: {title}\nData: {doc['text']}")
    context_string = "\n\n".join(context_blocks)
    
    prompt = PromptTemplate.from_template(template).format(
        context=context_string, 
        question=query
    )
    
    # Execute generation
    response = llm.invoke(prompt)
    return response.content


def get_movie_poster(movie_id: str, df: pd.DataFrame) -> Optional[str]:
    """Fetches a movie poster URL using the Pandas lookup fallback."""
    if not movie_id:
        return None
    try:
        poster_path = df[df['id'] == int(movie_id)]['poster_path'].values[0]
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        logger.warning(f"Could not fetch poster for movie ID {movie_id}: {e}")
        return None


# --- App Initialization ---
pc, vectorstore, llm, df = initialize_services()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

# --- UI Layout ---
st.title("CineMate AI Recommender")
st.markdown("<h5>Tell me what you're in the mood for, and I'll find the perfect movie for you!</h5>", unsafe_allow_html=True)

# Example Prompts
st.markdown("<p style='text-align: center; color: #E0F2F1; font-weight: 300;'>Try one of these prompts or write your own:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Christian Bale as batman and Heath Ledger as Joker", use_container_width=True):
        st.session_state.user_query = "Christian Bale as batman and Heath Ledger as Joker"
with col2:
    if st.button("A funny movie with talking animals", use_container_width=True):
        st.session_state.user_query = "A funny movie with talking animals"
with col3:
    if st.button("A sci-fi movie that makes you think", use_container_width=True):
        st.session_state.user_query = "A sci-fi movie that makes you think"


# Input Form
with st.form(key='recommendation_form'):
    user_query = st.text_input(
        "Your Movie Preference",
        value=st.session_state.user_query,
        placeholder="e.g., 'a mind-bending sci-fi movie with a twist ending'",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button(label='Get Recommendation ✨')


# --- Pipeline Execution Trigger ---
if submit_button and user_query:
    with st.spinner('Retrieving, Reranking, and Casting... 🍿'):
        try:
            # 1. Rerank Retrieval
            ranked_docs = retrieve_and_rerank(user_query, vectorstore, pc, top_k=5)      
            # 2. LLM Synthesis
            if not ranked_docs:
                recommendation = "Sorry, I couldn't find any relevant movies in our database for that request."
            else:
                recommendation = generate_llm_recommendation(user_query, ranked_docs, llm)
            
            # 3. State Management
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": recommendation})
            st.session_state.user_query = "" 
            st.rerun()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            st.error(f"An error occurred during generation: {e}")

# --- Conversation History Render ---
if st.session_state.messages:
    st.markdown("---")
    for message in reversed(st.session_state.messages): 
        if message["role"] == "assistant":
            recommendation_text = message['content']
            
            # Extract metadata using regex based on the LLM prompt structure
            title, year, movie_id = None, None, None
            title_match = re.search(r"\*\*Movie:\*\* (.*?) \((\d{4})\)", recommendation_text)
            if title_match:
                title, year = title_match.groups()

            id_match = re.search(r"\*\*ID\*\*: (\d+)", recommendation_text)
            if id_match:
                movie_id = id_match.group(1)

            # Display Recommendation Card
            with st.container():
                st.success("CineMate's Recommendation:")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if title and year:
                        poster_url = get_movie_poster(movie_id, df)
                        if poster_url:
                            st.image(poster_url, caption=f"{title} ({year})")
                        else:
                            st.info("No poster available.")
                with col2:
                    st.markdown(recommendation_text)