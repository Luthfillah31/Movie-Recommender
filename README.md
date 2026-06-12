# CineMate: AI Movie Recommender

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?style=flat-square)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=flat-square)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

CineMate is a conversational movie recommendation system built using a Retrieval-Augmented Generation (RAG) architecture. It allows users to query a vector database of films using natural language to retrieve personalized, context-aware recommendations based on genre, plot, director, or mood.

<img src="https://github.com/user-attachments/assets/69c69339-4134-4509-9e18-b85c4eca97e4" alt="CineMate Application Interface" width="800" style="border: 1px solid #333; border-radius: 6px;"/>

---

## Architecture & Technical Implementation

This project implements an advanced RAG pipeline to ensure high-accuracy retrieval and formatting:

1. **Query Processing:** The user's text input is simultaneously converted into a dense vector embedding (via OpenRouter) and a sparse keyword vector (via BM25).
2. **Hybrid Search:** Both vectors are queried against a Pinecone Vector Database using an `alpha=0.5` weighting, ensuring a balanced retrieval of semantic meaning and exact keyword matches (e.g., specific actors or directors).
3. **Cross-Encoder Reranking:** The initial retrieval payload is passed through a Pinecone Inference `bge-reranker-v2-m3` model to precision-rank the documents before they enter the LLM context window.
4. **LLM Synthesis:** The optimized context is passed to the LLM (`deepseek-v4-flash` via OpenRouter) using LangChain. The prompt strictly instructs the model to format the output consistently and avoid major plot spoilers.
5. **UI & Metadata:** The frontend, built in Streamlit, intercepts the Pinecone metadata to dynamically render official movie posters alongside the AI's structural response.

### Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Web application interface and session state management. |
| **Orchestration** | LangChain | Managing prompt templates and LLM integration. |
| **Vector Database** | Pinecone | Storing embeddings and executing Hybrid Search. |
| **Reranker** | Pinecone Inference | Cross-encoder precision ranking (`bge-reranker-v2-m3`). |
| **LLM & Embeddings**| OpenRouter | API gateway for DeepSeek generation and Perplexity dense embeddings. |
| **Sparse Encoder** | BM25 | Keyword tokenization for Pinecone sparse vector generation. |
| **Environment** | Pixi | Deterministic package management and CI/CD consistency. |

---

## Key Features

* **Conversational Interface:** Accepts unstructured, natural language inputs for movie discovery.
* **Hybrid Retrieval System:** Utilizes both semantic similarity and exact keyword matching to improve search accuracy over basic dense retrieval.
* **Automated Data Pipeline:** Includes a configured CI/CD workflow via GitHub Actions (`.github/workflows/update_data_pipeline.yml`) to schedule data updates.
* **Deterministic Environment:** Managed via `pixi.toml` and `pixi.lock` to guarantee dependency resolution and prevent package conflicts across deployments.

---

## Contact & Links

* **Developer:** Luthfillah Akhtar Fakhrudin (luthfillahatar@gmail.com)
* **Live Deployment:** [CineMate Recommender](https://movie-recommender-by-luthfillah.streamlit.app/)

## Acknowledgements

* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [Pinecone](https://www.pinecone.io/)
* [OpenRouter](https://openrouter.ai/)
