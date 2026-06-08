import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import requests
import streamlit as st
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
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
    page_title="CineMate AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Professional, muted "Slate and Steel" theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .st-emotion-cache-1y4p8pa { max-width: 850px; margin: 0 auto; }
    
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 2.8rem;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }

    h5 { 
        color: #94A3B8; 
        text-align: center; 
        font-weight: 400; 
        margin-bottom: 2.5rem; 
        font-size: 1.1rem;
    }
    
    /* Subtle outline buttons for example prompts */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #334155;
        background-color: transparent;
        color: #CBD5E1;
        transition: all 0.2s ease;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }

    .stButton>button:hover {
        background-color: #1E293B;
        color: #F8FAFC;
        border-color: #475569;
    }
    
    /* Clean, deep slate form */
    .stForm {
        background-color: #0F172A;
        border-radius: 12px;
        padding: 2rem; 
        border: 1px solid #1E293B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 1rem;
    }

    .stTextInput>div>div>input {
        border-radius: 8px;
        background-color: #1E293B;
        border: 1px solid #334155;
        color: #F8FAFC;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 1px #3B82F6;
    }
    
    /* Professional solid primary button */
    [data-testid="stFormSubmitButton"] {
        margin-top: 1.5rem;
    }

    [data-testid="stFormSubmitButton"] button {
        width: 100%; 
        font-weight: 600;
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        transition: background-color 0.2s ease;
    }
    
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #1D4ED8;
    }

    /* Movie Result Cards */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #0F172A;
        border-color: #1E293B !important;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


class OpenRouterEmbeddings(Embeddings):
    """Custom LangChain Embeddings wrapper for OpenRouter."""
    def __init__(self, api_key: str, model: str = "perplexity/pplx-embed-v1-0.6b"):
        self.api_key = api_key
        self.model = model
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
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

@st.cache_resource(show_spinner="Initializing AI Services...")
def initialize_services() -> Tuple[Pinecone, PineconeVectorStore, ChatOpenAI]:
    pinecone_key = st.secrets.get("PINECONE_API_KEY")
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
    index_name = st.secrets.get("PINECONE_INDEX_NAME", "movie-recommender")

    if not all([pinecone_key, openrouter_key]):
        st.error("Critical Error: Missing API keys in Streamlit Secrets.")
        st.stop()
        
    os.environ["PINECONE_API_KEY"] = pinecone_key

    try:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        embeddings = OpenRouterEmbeddings(api_key=openrouter_key)
        
        bm25 = BM25Encoder()
        bm25.load("bm25_model.json") 

        llm = ChatOpenAI(
            model_name="deepseek/deepseek-v4-flash",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=10000
        )

        return pc, index, embeddings, bm25, llm

    except Exception as e:
        logger.error(f"Service initialization failed: {e}", exc_info=True)
        st.error(f"Failed to connect to backend services: {e}")
        st.stop()


def retrieve_and_rerank(query: str, index: Any, embeddings: OpenRouterEmbeddings, bm25: BM25Encoder, pc: Pinecone, top_k: int = 5) -> List[Dict[str, Any]]:
    dense_vec = embeddings.embed_query(query)
    sparse_vec = bm25.encode_queries(query)
    
    hdense, hsparse = hybrid_scale(dense_vec, sparse_vec, alpha=0.5)
    
    initial_results = index.query(
        top_k=50,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )
    
    if not initial_results.matches:
        return []

    documents_payload = [
        {"id": match.id, "text": match.metadata["text"], "metadata": match.metadata} 
        for match in initial_results.matches
    ]
    
    reranked_results = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents_payload,
        top_n=top_k,
        return_documents=True
    )
    
    final_docs = []
    for match in reranked_results.data:
        final_docs.append({
            "text": match.document.text,
            "metadata": match.document.metadata,
            "score": match.score
        })
        
    return final_docs

def hybrid_scale(dense: List[float], sparse: Dict[str, List[float]], alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def generate_llm_recommendation(query: str, ranked_docs: List[Dict[str, Any]], llm: ChatOpenAI, num_recs: int = 1) -> str:
    template = """
    You are 'CineMate', a friendly and enthusiastic movie expert chatbot. 
    CRITICAL INSTRUCTION: You MUST provide EXACTLY {num_recs} movie recommendation(s).

    **Your Instructions:**
    1. Analyze the user's request.
    2. Use the provided **Context from our movie database** to find the best matches.
    3. Crucially, DO NOT spoil major plot twists or endings.
    4. You MUST separate every single movie recommendation with this exact string on its own line: ---SPLIT---
    5. Present EACH recommendation in the exact structured format below.

    **Context from our movie database:**
    {context}

    **User's Request:**
    {question}

    **FORMAT SHOULD LOOK LIKE THIS (Repeat EXACTLY {num_recs} times, separated by ---SPLIT---):**
    
    **Movie:** [Movie Title] ([Year])
    **Logline:** [A compelling, one-sentence hook]

    **Synopsis:** [A brief, 2-3 sentence summary of the plot]

    **Why You'll Like It:** [Directly connect the movie to the user's stated preferences]
    **Details:**
    * **Genre:** [Primary Genre(s)]
    * **Director:** [Director's Name]
    * **Starring:** [Lead Actor(s)]
    """
    
    context_blocks = []
    for doc in ranked_docs:
        title = doc["metadata"].get("title", "Unknown Title")
        context_blocks.append(f"Title: {title}\nData: {doc['text']}")
    context_string = "\n\n".join(context_blocks)
    
    prompt = PromptTemplate.from_template(template).format(
        context=context_string, question=query, num_recs=num_recs
    )
    
    response = llm.invoke(prompt)
    return response.content

def get_movie_poster(title: str) -> Optional[str]:
    if not title:
        return None
    poster_path = st.session_state.posters.get(title)
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None


# --- App Initialization ---
pc, index, embeddings, bm25, llm = initialize_services()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'posters' not in st.session_state:
    st.session_state.posters = {}


# --- UI Layout ---
st.title("CineMate AI")
st.markdown("<h5>Tell me what you're in the mood for, and I'll find the perfect movie.</h5>", unsafe_allow_html=True)

# Example Prompts
colA, colB, colC = st.columns(3)
with colA:
    if st.button("A mind-bending thriller by Nolan", use_container_width=True):
        st.session_state.user_query = "A mind-bending thriller by Christopher Nolan"
with colB:
    if st.button("An inspiring marathon documentary", use_container_width=True):
        st.session_state.user_query = "An inspiring documentary about marathon runners"
with colC:
    if st.button("A tactical espionage action movie", use_container_width=True):
        st.session_state.user_query = "A tense, tactical espionage action movie"


# Standard, Professional Stacked Form
with st.form(key='recommendation_form'):
    st.markdown("<p style='color: #E2E8F0; font-weight: 600; margin-bottom: 0.5rem;'>Movie Preferences</p>", unsafe_allow_html=True)
    user_query = st.text_input(
        "Query",
        value=st.session_state.user_query,
        placeholder="e.g., 'A gripping psychological thriller set in space'",
        label_visibility="collapsed"
    )
    
    st.markdown("<p style='color: #E2E8F0; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;'>Results to generate</p>", unsafe_allow_html=True)
    num_recs = st.selectbox(
        "Count", 
        options=[1, 2, 3, 4, 5], 
        index=2,
        label_visibility="collapsed"
    )
    
    submit_button = st.form_submit_button(label='Search Movies', use_container_width=True)


# --- Pipeline Execution ---
if submit_button and user_query:
    st.session_state.messages = [] 
    
    with st.spinner(f'Searching our vector database for the top matches...'):
        try:
            ranked_docs = retrieve_and_rerank(user_query, index, embeddings, bm25, pc, top_k=max(10, num_recs * 5))
            
            for doc in ranked_docs:
                title = doc["metadata"].get("title")
                path = doc["metadata"].get("poster_path")
                if title and path:
                    st.session_state.posters[title] = path

            if not ranked_docs:
                recommendation = "Sorry, I couldn't find any relevant movies in our database for that request."
            else:
                recommendation = generate_llm_recommendation(user_query, ranked_docs, llm, num_recs)
            
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": recommendation})
            st.session_state.user_query = "" 
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            st.error(f"An error occurred during generation: {e}")


# --- Results Render (Once only, safely below the form) ---
if st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    
    for message in st.session_state.messages: 
        if message["role"] == "assistant":
            recommendation_text = message['content']
            movie_blocks = recommendation_text.split("---SPLIT---")
            
            for block in movie_blocks:
                if not block.strip():
                    continue 
                
                title, year = None, None
                title_match = re.search(r"\*\*Movie:\*\* (.*?) \((\d{4})\)", block)
                if title_match:
                    title, year = title_match.groups()
                    title = title.strip()

                with st.container(border=True):
                    if title and year:
                        st.markdown(f"<h2 style='color: #F8FAFC; margin-bottom: 1rem;'>🎬 {title} ({year})</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h2 style='color: #F8FAFC; margin-bottom: 1rem;'>🎬 Recommended Movie</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 3]) 
                    with col1:
                        if title and year:
                            poster_url = get_movie_poster(title)
                            if poster_url:
                                st.image(poster_url, use_container_width=True) 
                            else:
                                st.info("No poster available.")
                    with col2:
                        clean_text = re.sub(r"\*\*Movie:\*\* .*?\(\d{4}\)\n*", "", block)
                        st.markdown(clean_text)