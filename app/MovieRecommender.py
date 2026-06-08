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

@st.cache_resource(show_spinner="Initializing AI Services...")
def initialize_services() -> Tuple[Pinecone, PineconeVectorStore, ChatOpenAI]:
    """Idempotent initialization of API clients and Langchain wrappers."""
    
    pinecone_key = st.secrets.get("PINECONE_API_KEY")
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
    index_name = st.secrets.get("PINECONE_INDEX_NAME", "movie-recommender")

    if not all([pinecone_key, openrouter_key]):
        st.error("Critical Error: Missing API keys in Streamlit Secrets.")
        st.stop()
        
    os.environ["PINECONE_API_KEY"] = pinecone_key

    try:
        # 2. Raw Pinecone Client
        pc = Pinecone(api_key=pinecone_key)
        
        # Initialize the specific Pinecone Index directly
        index = pc.Index(index_name)

        # 3. Embeddings & BM25 Encoder (The Hybrid Duo)
        embeddings = OpenRouterEmbeddings(api_key=openrouter_key)
        
        bm25 = BM25Encoder()
        bm25.load("bm25_model.json") # Load your pipeline's keyword vocabulary

        # 4. Langchain LLM (Generative)
        llm = ChatOpenAI(
            model_name="deepseek/deepseek-v4-flash",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=10000
        )

        # Notice we return 'index' and 'bm25' now instead of 'vectorstore'
        return pc, index, embeddings, bm25, llm

    except Exception as e:
        logger.error(f"Service initialization failed: {e}", exc_info=True)
        st.error(f"Failed to connect to backend services: {e}")
        st.stop()

# --- Pipeline Execution Functions ---
def retrieve_and_rerank(query: str, index: Any, embeddings: OpenRouterEmbeddings, bm25: BM25Encoder, pc: Pinecone, top_k: int = 5) -> List[Dict[str, Any]]:
    """Executes a 50/50 Hybrid Search followed by a precision Cross-Encoder rerank."""
    
    # Step 1: Generate both Dense and Sparse Vectors for the user's query
    dense_vec = embeddings.embed_query(query)
    sparse_vec = bm25.encode_queries(query)
    
    # Step 2: Apply the Hybrid Weighting (0.5 = 50% semantic, 50% keyword)
    hdense, hsparse = hybrid_scale(dense_vec, sparse_vec, alpha=0.5)
    
    # Step 3: Execute Broad Hybrid Fetch (Get top 50 candidates)
    initial_results = index.query(
        top_k=50,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )
    
    if not initial_results.matches:
        return []

    # Step 4: Format payload for Pinecone Inference Reranker
    documents_payload = [
        {"id": match.id, "text": match.metadata["text"], "metadata": match.metadata} 
        for match in initial_results.matches
    ]
    
    # Step 5: Execute Cross-Encoder Reranking
    reranked_results = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents_payload,
        top_n=top_k,
        return_documents=True
    )
    
    # Step 6: Parse final ranked payload
    final_docs = []
    for match in reranked_results.data:
        final_docs.append({
            "text": match.document.text,
            "metadata": match.document.metadata,
            "score": match.score
        })
        
    return final_docs

def hybrid_scale(dense: List[float], sparse: Dict[str, List[float]], alpha: float):
    """
    Scales vectors for hybrid search: 
    alpha = 1.0 (pure semantic), alpha = 0.0 (pure keyword), alpha = 0.5 (equal blend)
    """
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

    **FORMAT SHOULD LOOKS LIKE THIS (Repeat EXACTLY {num_recs} times, separated by ---SPLIT---):**
    
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
    """Fetches a movie poster URL directly from the Pinecone metadata saved in memory."""
    if not title:
        return None
    # Lookup the exact title generated by the LLM in our cached dictionary
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
st.title("🍿 CineMate AI Recommender")
st.markdown("<h5 style='text-align: center; color: #E0F2F1; margin-bottom: 2rem;'>Tell me what you're in the mood for, and I'll find the perfect movie!</h5>", unsafe_allow_html=True)

# Example Prompts
st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 0.9rem;'>Try one of these prompts or write your own:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("A mind-bending thriller by Christopher Nolan", use_container_width=True):
        st.session_state.user_query = "A mind-bending thriller by Christopher Nolan"
with col2:
    if st.button("An inspiring documentary about marathon runners", use_container_width=True):
        st.session_state.user_query = "An inspiring documentary about marathon runners"
with col3:
    if st.button("A tense, tactical espionage action movie", use_container_width=True):
        st.session_state.user_query = "A tense, tactical espionage action movie"


# Input Form
with st.form(key='recommendation_form'):
    col1, col2 = st.columns([3, 1]) # 3-to-1 width ratio
    
    with col1:
        user_query = st.text_input(
            "Your Movie Preference",
            value=st.session_state.user_query,
            placeholder="e.g., 'a mind-bending sci-fi movie with a twist ending'",
            label_visibility="collapsed"
        )
    with col2:
        # Replaced the slider with a clean dropdown box
        num_recs = st.selectbox(
            "Recommendations", 
            options=[1, 2, 3, 4, 5], 
            index=2, # Default to 3
            label_visibility="collapsed"
        )
        
    submit_button = st.form_submit_button(label='Get Recommendation ✨')


# --- Pipeline Execution Trigger ---
if submit_button and user_query:
    st.session_state.messages = [] 
    
    with st.spinner(f'Retrieving, Reranking, and Casting {num_recs} movies... 🍿'):
        try:
            ranked_docs = retrieve_and_rerank(user_query, index, embeddings, bm25, pc, top_k=max(10, num_recs * 5))
            # MEMORY SAVE: Cache the poster paths from Pinecone directly into session state
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
            st.rerun()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            st.error(f"An error occurred during generation: {e}")

# --- Conversation History Render ---
if st.session_state.messages:
    st.markdown("---")
    for message in st.session_state.messages: 
        if message["role"] == "assistant":
            recommendation_text = message['content']
            movie_blocks = recommendation_text.split("---SPLIT---")
            
            for block in movie_blocks:
                if not block.strip():
                    continue 
                
                # Extract the Title directly
                title, year = None, None
                title_match = re.search(r"\*\*Movie:\*\* (.*?) \((\d{4})\)", block)
                if title_match:
                    title, year = title_match.groups()
                    title = title.strip() # Ensure clean matching

                with st.container(border=True):
                    if title and year:
                        st.markdown(f"## 🎬 {title} ({year})")
                    else:
                        st.markdown("## 🎬 CineMate's Recommendation")
                    
                    st.write("") 
                    
                    col1, col2 = st.columns([1, 2.5]) 
                    with col1:
                        if title and year:
                            # Pass the Title to get the Pinecone metadata URL
                            poster_url = get_movie_poster(title)
                            if poster_url:
                                st.image(poster_url, use_container_width=True) 
                            else:
                                st.info("No poster available.")
                    with col2:
                        # Clean the raw text output to remove the redundant Title line
                        clean_text = re.sub(r"\*\*Movie:\*\* .*?\(\d{4}\)\n*", "", block)
                        st.markdown(clean_text)