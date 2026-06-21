import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

import requests
import streamlit as st
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pydantic import BaseModel, Field
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CineMate AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMainBlockContainer"] {
        max-width: 850px;
        margin: 0 auto;
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
</style>
""", unsafe_allow_html=True)


class MovieRecommendation(BaseModel):
    title: str = Field(description="The exact movie title")
    year: str = Field(description="The release year")
    logline: str = Field(description="A compelling, one-sentence hook")
    synopsis: str = Field(description="A brief, 2-3 sentence plot summary")
    reasoning: str = Field(description="Why the user will like it based on their preferences")
    genres: str = Field(description="Primary genres")
    director: str = Field(description="Director's name")
    starring: str = Field(description="Lead actors")


class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]


def generate_embedding(text: str, api_key: str, model: str = "perplexity/pplx-embed-v1-0.6b") -> List[float]:
    """Generate a dense vector embedding using OpenRouter's API."""
    response = requests.post(
        url="https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={"model": model, "input": text}
    )
    if response.status_code != 200:
        raise ValueError(f"OpenRouter embedding failed: {response.text}")
    return response.json()["data"][0]["embedding"]


@st.cache_resource(show_spinner="Initializing AI Services...")
def initialize_services() -> Tuple[Pinecone, Any, BM25Encoder, OpenAI]:
    """Initialize Pinecone, BM25 encoder, and the OpenAI client."""
    pinecone_key = st.secrets.get("PINECONE_API_KEY")
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
    index_name = st.secrets.get("PINECONE_INDEX_NAME", "movie-recommender")

    if not all([pinecone_key, openrouter_key]):
        st.error("Missing required API keys in Streamlit secrets.")
        st.stop()

    try:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        
        bm25 = BM25Encoder()
        bm25.load("bm25_model.json") 

        llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )

        return pc, index, bm25, llm_client
    except Exception as e:
        logger.error(f"Service initialization failed: {e}", exc_info=True)
        st.error("Failed to connect to backend services.")
        st.stop()


def retrieve_and_rerank(query: str, index: Any, bm25: BM25Encoder, pc: Pinecone, openrouter_key: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top matches from Pinecone hybrid search and rerank them using BGE."""
    dense_vec = generate_embedding(query, openrouter_key)
    sparse_vec = bm25.encode_queries(query)
    
    alpha = 0.5
    hsparse = {
        'indices': sparse_vec['indices'],
        'values':  [v * (1 - alpha) for v in sparse_vec['values']]
    }
    hdense = [v * alpha for v in dense_vec]
    
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
    
    return [
        {
            "text": match.document.get("text") if isinstance(match.document, dict) else match.document.text,
            "metadata": match.document.get("metadata") if isinstance(match.document, dict) else match.document.metadata,
            "score": match.score
        }
        for match in reranked_results.data
    ]


def generate_recommendations(query: str, ranked_docs: List[Dict[str, Any]], llm_client: OpenAI, num_recs: int) -> RecommendationResponse:
    """Generate structured movie recommendations using the LLM based on retrieved context."""
    context_blocks = [
        f"Title: {doc['metadata'].get('title', 'Unknown')}\nData: {doc['text']}"
        for doc in ranked_docs
    ]
    context_string = "\n\n".join(context_blocks)
    
    system_prompt = (
        "You are 'CineMate', an expert movie recommendation assistant.\n"
        "Analyze the user's request and recommend movies from the provided context database.\n"
        f"You must return EXACTLY {num_recs} recommendation(s).\n"
        "Do not spoil major plot twists.\n"
        "You MUST return valid JSON that perfectly adheres to the following JSON schema:\n"
        f"{json.dumps(RecommendationResponse.model_json_schema(), indent=2)}"
    )
    
    user_prompt = f"Context:\n{context_string}\n\nUser's Request:\n{query}"

    response = llm_client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )

    result_text = response.choices[0].message.content
    return RecommendationResponse.model_validate_json(result_text)


def get_movie_poster(title: str) -> Optional[str]:
    """Retrieve the TMDB poster URL if available."""
    if title and title in st.session_state.posters:
        poster_path = st.session_state.posters[title]
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None


def main():
    pc, index, bm25, llm_client = initialize_services()

    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'posters' not in st.session_state:
        st.session_state.posters = {}

    st.markdown("<h1 style='text-align: center; font-weight: 700; font-size: 2.8rem; color: #F8FAFC; margin-bottom: 0.5rem;'>CineMate AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #94A3B8; font-weight: 400; margin-bottom: 2rem;'>Tell me what you're in the mood for, and I'll find the perfect movie.</h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    if col1.button("A mind-bending thriller by Nolan", width="stretch"):
        st.session_state.user_query = "A mind-bending thriller by Christopher Nolan"
    if col2.button("An inspiring marathon documentary", width="stretch"):
        st.session_state.user_query = "An inspiring documentary about marathon runners"
    if col3.button("A tactical espionage action movie", width="stretch"):
        st.session_state.user_query = "A tense, tactical espionage action movie"

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
        
        submit_button = st.form_submit_button(label='Search Movies', width="stretch")

    if submit_button and user_query:
        st.session_state.recommendations = None 
        
        with st.spinner('Searching our vector database for the top matches...'):
            try:
                openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
                ranked_docs = retrieve_and_rerank(user_query, index, bm25, pc, openrouter_key, top_k=max(10, num_recs * 5))
                
                for doc in ranked_docs:
                    title = doc["metadata"].get("title")
                    path = doc["metadata"].get("poster_path")
                    if title and path:
                        st.session_state.posters[title] = path

                if not ranked_docs:
                    st.warning("Sorry, I couldn't find any relevant movies in our database for that request.")
                else:
                    recs = generate_recommendations(user_query, ranked_docs, llm_client, num_recs)
                    st.session_state.recommendations = recs.recommendations
                
                st.session_state.user_query = ""
                
            except Exception as e:
                logger.error(f"Generation failed: {e}", exc_info=True)
                st.error("An error occurred while generating recommendations.")

    if st.session_state.recommendations:
        st.markdown("<br>", unsafe_allow_html=True)
        for rec in st.session_state.recommendations:
            with st.container(border=True):
                st.markdown(f"<h2 style='color: #F8FAFC; margin-bottom: 1rem;'>🎬 {rec.title} ({rec.year})</h2>", unsafe_allow_html=True)
                
                rc1, rc2 = st.columns([1, 3]) 
                with rc1:
                    poster_url = get_movie_poster(rec.title)
                    if poster_url:
                        st.image(poster_url, width="stretch") 
                    else:
                        st.info("No poster available.")
                with rc2:
                    st.markdown(f"**Logline:** {rec.logline}")
                    st.markdown(f"**Synopsis:** {rec.synopsis}")
                    st.markdown(f"**Why You'll Like It:** {rec.reasoning}")
                    st.markdown("---")
                    st.markdown(f"**Genre:** {rec.genres} &nbsp;|&nbsp; **Director:** {rec.director} &nbsp;|&nbsp; **Starring:** {rec.starring}")


if __name__ == "__main__":
    main()