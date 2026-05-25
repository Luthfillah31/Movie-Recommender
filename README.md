<h1 align="center">🎬 CineMate AI Recommender ✨</h1>

<p align="center">
  <strong>Your AI-powered movie night savior! Chat with CineMate to discover the perfect film based on your mood.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-red?style=for-the-badge&logo=streamlit" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/LangChain-0.1%2B-green?style=for-the-badge" alt="LangChain Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <em>Tired of endlessly scrolling through streaming services? CineMate is an intelligent chatbot that uses the power of Large Language Models to give you personalized movie recommendations. Just tell it what you're looking for, and get a great suggestion in seconds!</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/69c69339-4134-4509-9e18-b85c4eca97e4" alt="CineMate Application Screenshot" style="border-radius: 10px;"/>
  <br>
</p>

---

## 🌟 Key Features

-   **💬 Conversational Interface:** Chat with a friendly AI movie expert in natural language.
-   **🧠 Intelligent Recommendations:** Powered by a Retrieval-Augmented Generation (RAG) pipeline for relevant, context-aware suggestions.
-   **🖼️ Movie Posters:** See the official movie poster for every recommendation, fetched in real-time.
-   **🚫 Spoiler-Free:** Get detailed synopses without ruining the movie's twists.
-   **🎨 Sleek & Modern UI:** A beautiful, responsive interface built with Streamlit and custom CSS.

---

## 🛠️ How It Works: The Tech Stack

CineMate uses a modern RAG (Retrieval-Augmented Generation) architecture to provide high-quality recommendations.

1.  **Embedding:** The user's query is converted into a vector embedding using a `HuggingFace Sentence-Transformer` model.
2.  **Retrieval:** This vector is used to search a `Pinecone` vector database, which contains thousands of movies, to find the most semantically similar films.
3.  **Augmentation:** The metadata of the retrieved movies (title, synopsis, genre, etc.) is collected.
4.  **Generation:** This context, along with the original query, is passed to the `Mistral` language model via a `LangChain` prompt. The LLM then generates a human-like, structured recommendation based *only* on the provided data.

| Component      | Technology                                                                                                  | Purpose                                       |
| -------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| **Frontend** | <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit" alt="Streamlit">            | Building the interactive web UI               |
| **LLM Chain** | <img src="https://img.shields.io/badge/LangChain-008661?style=flat" alt="LangChain">                           | Orchestrating the RAG pipeline                |
| **LLM** | <img src="https://img.shields.io/badge/OpenRouter-8A2BE2?style=flat" alt="OpenRouter">                         | Accessing the Mistral model                   |
| **Embeddings** | <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface" alt="Hugging Face">    | Text-to-vector conversion                     |
| **Vector DB** | <img src="https://img.shields.io/badge/Pinecone-008080?style=flat&logo=pinecone" alt="Pinecone">               | Storing and searching movie vectors           |
| **Movie Data** | <img src="https://img.shields.io/badge/TMDB-01B4E4?style=flat" alt="TMDB">                                     | Fetching movie posters via API                |

---

## 🙏 Acknowledgements

-   [Streamlit](https://streamlit.io/) for the amazing web framework.
-   [LangChain](https://www.langchain.com/) for making complex LLM workflows simple.
-   [Pinecone](https://www.pinecone.io/) for the powerful vector database.
-   [Hugging Face](https://huggingface.co/) for the sentence-transformer models.
-   [The Movie Database (TMDB)](https://www.themoviedb.org/) for their free movie data API.

---

## 📧 Contact

luthfillahatar@gmail.com

Project Link: https://movie-recommender-by-luthfillah.streamlit.app/
