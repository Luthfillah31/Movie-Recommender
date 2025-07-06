import streamlit as st
import os
import re
import pandas as pd
# --- LangChain Imports ---
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Page Configuration ---
st.set_page_config(
    page_title="CineMate AI Recommender",
    page_icon="🎥",  # Changed page icon
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Enhanced "Blue Water" Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    /* --- Center the main content --- */
    .st-emotion-cache-1y4p8pa {
        max-width: 900px;
    }
    
    /* --- Main Title & Subheader --- */
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 3rem; /* Slightly larger font for title */
        background: linear-gradient(45deg, #1E40AF, #58C4E8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    h5 {
        color: #E0F2F1;
        text-align: center;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* --- Example Prompt Buttons --- */
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
    
    /* --- Main Form & Input (Enhanced) --- */
    .stForm {
        background-image: linear-gradient(to bottom right, rgba(22, 58, 92, 0.7), rgba(13, 33, 55, 0.7));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem 3rem; /* Increased padding */
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-top: 2rem;
        transition: all 0.3s ease-in-out; /* Added transition for hover effect */
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
    
    /* --- Submit Button in Form (Enhanced & Centered) --- */
    [data-testid="stFormSubmitButton"] {
        display: flex;
        justify-content: center; /* Center the button */
        padding-top: 1rem; /* Add some space above the button */
    }

    [data-testid="stFormSubmitButton"] button {
        width: 60%; /* Set a specific width */
        font-weight: 700;
        background: linear-gradient(45deg, #00a9ff, #58C4E8); /* Added gradient */
        color: white;
        border: none;
        box-shadow: 0 4px 15px 0 rgba(0, 169, 255, 0.3);
        transition: all 0.3s ease-in-out;
    }
    
    [data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 6px 25px 0 rgba(0, 169, 255, 0.5);
        transform: translateY(-3px);
    }

    /* --- Conversation History & Messages --- */
    /* User Message */
    [data-testid="stNotification"][data-st-notification-status="info"] {
        background: rgba(88, 196, 232, 0.2);
        border: 1px solid rgba(88, 196, 232, 0.5);
        border-radius: 15px;
        color: #E0F2F1;
    }

    /* Assistant Message */
    [data-testid="stNotification"][data-st-notification-status="success"] {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: #E0F2F1;
    }
    
    /* --- Expander for Source Docs --- */
    [data-testid="stExpander"] {
        background: rgba(22, 58, 92, 0.4);
        border: 1px solid rgba(88, 196, 232, 0.5);
        border-radius: 15px;
        margin-top: 1rem;
    }

    [data-testid="stExpander"] summary {
        color: #E0F2F1;
        font-weight: 400;
    }
    
    [data-testid="stExpander"] summary:hover {
        color: #58C4E8;
    }
</style>
""", unsafe_allow_html=True)


# --- API and Service Initialization ---
try:
    # --- API Keys ---
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
    openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY")
    tmdb_api_key = st.secrets.get("TMDB_API_KEY") # For fetching movie posters

    if not all([pinecone_api_key, openrouter_api_key, tmdb_api_key]):
        st.error("Please set PINECONE_API_KEY, OPENROUTER_API_KEY, and TMDB_API_KEY in your Streamlit secrets.")
        st.stop()
        
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    PINECONE_INDEX_NAME = "movie-recommender" # <--- REPLACE WITH YOUR INDEX NAME

    # --- LangChain Component Initialization ---
    @st.cache_resource
    def load_models():
        embeddings = HuggingFaceEmbeddings(model_name='./all-MiniLM-L6-v2-download')
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        llm = ChatOpenAI(
            model_name="mistralai/mistral-small-3.2-24b-instruct:free",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=500
        )

        df = pd.read_csv('MovieData.csv')
        return embeddings, vectorstore, llm, df

    embeddings, vectorstore, llm, df = load_models()

    template = """
    You are 'CineMate', a friendly and enthusiastic movie expert chatbot. Your goal is to give excellent, personalized movie recommendations by using the provided movie data to answer the user's request.

    **Your Instructions:**
    1.  Analyze the user's request to understand their taste (genres, actors, directors, themes, etc.).
    2.  Use the provided **Context from our movie database** to find a movie that is a great match.
    3.  If you cannot find a good match in the context, say so and suggest that you can search more broadly if they'd like.
    4.  **Crucially, DO NOT spoil major plot twists or endings.**
    5.  Present your recommendation in the structured format below, extracting the information directly from the provided context.

    **Context from our movie database:**
    {context}

    **User's Request:**
    {question}

    **Your Recommendation:**
    
    **ID**: [Movie ID]
    
    **Movie:** [Movie Title] ([Year])

    **Logline:** [A compelling, one-sentence hook to grab their attention, extracted from the context.]

    **Synopsis:** [A brief, 2-3 sentence summary of the plot without giving too much away, extracted from the context.]

    **Why You'll Like It:** [Directly connect the movie to the user's stated preferences from their request. For example: "Based on your request for a mind-bending sci-fi movie, this seems like a perfect fit because..."]

    **Details:**
    * **Genre:** [Primary Genre(s)]
    * **Director:** [Director's Name]
    * **Starring:** [Lead Actor(s)]

    [Ask a follow-up question to keep the conversation going, like "Does that sound like something you'd be interested in?" or "Would you like another suggestion based on these details?"]
    """
    
    custom_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 10}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': custom_prompt}
    )

except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    st.stop()


# --- Helper Function for Movie Poster ---
def get_movie_poster(movie_id):
    """Fetches a movie poster URL from TMDB using the movie ID."""
    if not movie_id:
        return None
    try:
        poster_path = df[df['id'] == int(movie_id)]['poster_path'].values[0]
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Could not fetch poster for movie ID {movie_id}: {e}")
    return None

# --- Session State Initialization ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

# --- UI Components ---
st.title("CineMate AI Recommender")
st.markdown("<h5>Tell me what you're in the mood for, and I'll find the perfect movie for you!</h5>", unsafe_allow_html=True)

# --- Example Prompts ---
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


# --- Input Form ---
with st.form(key='recommendation_form'):
    user_query = st.text_input(
        "Your Movie Preference",
        value=st.session_state.user_query,
        placeholder="e.g., 'a mind-bending sci-fi movie with a twist ending'",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button(label='Get Recommendation ✨')


# --- RAG Logic and Display ---
if submit_button and user_query:
    with st.spinner('Casting for the perfect movie... 🍿'):
        try:
            result = rag_chain.invoke(user_query)
            recommendation = result.get('result', "Sorry, I couldn't find a recommendation based on that.")
            source_docs = result.get('source_documents')
            
            # --- Store message for history ---
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": recommendation, "source_docs": source_docs})
            st.session_state.user_query = "" # Clear input after submission
            st.rerun() # Rerun to display the new message immediately

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Display Conversation History ---
if st.session_state.messages:
    st.markdown("---")
    for message in reversed(st.session_state.messages): 
        if message["role"] == "assistant":
            recommendation_text = message['content']
            
            # --- Extract Title, Year, and ID for Poster ---
            title, year, movie_id = None, None, None
            title_match = re.search(r"\*\*Movie:\*\* (.*?) \((\d{4})\)", recommendation_text)
            if title_match:
                title, year = title_match.groups()

            id_match = re.search(r"\*\*ID\*\*: (\d+)", recommendation_text)
            if id_match:
                movie_id = id_match.group(1)

            # --- Display Recommendation with Poster ---
            with st.container():
                st.success("CineMate's Recommendation:")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if title and year:
                        poster_url = get_movie_poster(movie_id)
                        if poster_url:
                            st.image(poster_url, caption=f"Poster for {title}")
                        else:
                            st.write("(No poster found)")
                with col2:
                    st.markdown(recommendation_text)

            
