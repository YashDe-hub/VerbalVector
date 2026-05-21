"""
Manages the vector database (ChromaDB) interactions for storing and retrieving
transcript chunks and their embeddings.
"""

import logging
import time
import nltk
import chromadb
from chromadb.utils import embedding_functions

import config

logger = logging.getLogger(__name__)

# --- Constants ---
VECTOR_DB_PATH = str(config.VECTOR_DB_DIR)
COLLECTION_NAME = "transcripts"
_EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL
DEFAULT_EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL_NAME)

# --- Initialization ---
_chroma_client = None
_embedding_function = None
_collection_cache: dict = {}

def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB client (persistent path: {VECTOR_DB_PATH})...")
        try:
            _chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            logger.info("ChromaDB client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise # Re-raise the exception to signal failure
    return _chroma_client

def get_embedding_function():
    """Returns the embedding function/model instance."""
    global _embedding_function
    if _embedding_function is None:
        logger.info(f"Using ChromaDB SentenceTransformer embedding function ({_EMBEDDING_MODEL_NAME}).")
        _embedding_function = DEFAULT_EF
    return _embedding_function

def initialize_vector_store(collection_name: str = COLLECTION_NAME):
    """
    Returns the ChromaDB collection, creating it on first call.
    Subsequent calls return the cached collection object.
    """
    if collection_name in _collection_cache:
        return _collection_cache[collection_name]

    logger.info(f"Initializing vector store collection: '{collection_name}'...")
    try:
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            logger.info("Downloading NLTK 'punkt' tokenizer...")
            nltk.download('punkt', quiet=True)
        except Exception as nltk_e:
             logger.warning(f"Could not find or download NLTK 'punkt': {nltk_e}. Sentence tokenization might fail.")

        client = get_chroma_client()
        if not client:
            return None

        embedding_func = get_embedding_function()
        if not embedding_func:
             logger.error("Failed to get embedding function.")
             return None

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"},
        )
        _collection_cache[collection_name] = collection
        logger.info(f"Vector store collection '{collection_name}' ready.")
        return collection

    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
        return None

# --- Store Transcript Implementation ---
def store_transcript(
    transcript_text: str,
    source_id: str,
    collection,
    session_label: str = "",
    utterances: list[dict] | None = None,  # NEW
):
    """
    Chunks the transcript, generates embeddings (implicitly via collection),
    and stores them in the collection.

    When `utterances` is provided, chunks per-utterance and attaches the
    speaker ID to each chunk's metadata. Otherwise falls back to NLTK
    sentence chunking with no speaker metadata.
    """
    if not transcript_text or not source_id or not collection:
        logger.error("store_transcript called with invalid arguments.")
        return False

    logger.info(f"Storing transcript for source_id: '{source_id}'...")

    try:
        # Build chunks + per-chunk speakers
        if utterances:
            chunks = [u["text"] for u in utterances if u.get("text")]
            speakers = [u.get("speaker") for u in utterances if u.get("text")]
            logger.info(f"Chunking by {len(chunks)} utterance(s) with diarization.")
        else:
            try:
                sentences = nltk.sent_tokenize(transcript_text)
            except LookupError:
                logger.warning("NLTK 'punkt' tokenizer not found. Downloading...")
                nltk.download('punkt', quiet=True)
                sentences = nltk.sent_tokenize(transcript_text)
            except Exception as e:
                logger.error(f"Failed to tokenize transcript: {e}")
                return False

            if not sentences:
                logger.warning(f"Transcript for '{source_id}' resulted in zero sentences.")
                return False

            chunks = sentences
            speakers = [None] * len(chunks)
            logger.info(f"Split transcript into {len(sentences)} sentence chunks.")

        if not chunks:
            return False

        # Create IDs and metadata
        ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        metadata = []
        ts = time.time()
        for i in range(len(chunks)):
            entry = {
                "source": source_id,
                "chunk_index": i,
                "timestamp": ts,
                "session_label": session_label,
            }
            if speakers[i] is not None:
                entry["speaker"] = speakers[i]
            metadata.append(entry)

        logger.info(f"Adding {len(chunks)} chunks to collection '{collection.name}'...")
        collection.add(
            embeddings=None,
            documents=chunks,
            metadatas=metadata,
            ids=ids,
        )
        logger.info(f"Successfully stored transcript chunks for '{source_id}'.")
        return True

    except Exception as e:
        logger.error(f"Failed to store transcript for '{source_id}': {e}", exc_info=True)
        return False

# --- Search Transcripts Implementation ---
def search_transcripts(query: str, collection, n_results: int = 5, source_id: str | None = None) -> list[dict]:
    """
    Search the collection for relevant document chunks.

    Returns a list of dicts: {"text": str, "source_id": str, "session_label": str}
    """
    if not query or not collection:
        logger.error("search_transcripts called with invalid arguments.")
        return []

    logger.info(f"Searching collection '{collection.name}' for query: '{query:.50}...' (n_results={n_results})")

    try:
        query_kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas"],
        }
        if source_id:
            query_kwargs["where"] = {"source": source_id}

        results = collection.query(**query_kwargs)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        return [
            {
                "text": doc,
                "source_id": meta.get("source", ""),
                "session_label": meta.get("session_label", ""),
            }
            for doc, meta in zip(docs, metas)
        ]

    except Exception as e:
        logger.error(f"Failed to search transcripts: {e}", exc_info=True)
        return []

def list_sessions(collection) -> list[dict]:
    """Return a list of unique sessions stored in the collection."""
    if not collection:
        return []

    try:
        all_metadata = collection.get(include=["metadatas"])
        metadatas = all_metadata.get("metadatas", [])

        sessions: dict[str, dict] = {}
        for meta in metadatas:
            source = meta.get("source", "")
            if source not in sessions:
                sessions[source] = {
                    "source_id": source,
                    "session_label": meta.get("session_label", ""),
                    "timestamp": meta.get("timestamp", 0),
                    "chunk_count": 0,
                }
            sessions[source]["chunk_count"] += 1

        return sorted(sessions.values(), key=lambda s: s["timestamp"], reverse=True)

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        return []


if __name__ == '__main__':
    # Example Usage/Test
    logger.info("Running vector store manager test...")
    test_collection = initialize_vector_store()
    if test_collection:
        logger.info(f"Successfully initialized test collection: {test_collection.name}")
        # Add dummy calls once functions are implemented
        test_transcript = "This is the first test sentence. It talks about testing. This is the second sentence; it mentions vectors. The third sentence is about storage."
        source = "test_doc_main"
        logger.info("Attempting to store test transcript...")
        stored = store_transcript(test_transcript, source, test_collection)
        if stored:
             logger.info("Store successful. Attempting test search...")
             search_query = "What is mentioned about vectors?"
             results = search_transcripts(search_query, test_collection, n_results=2)
             logger.info(f"Test search results for '{search_query}':")
             for i, res in enumerate(results):
                 print(f"  Result {i+1}: {res}")
        else:
             logger.error("Store failed during test.")
            
    else:
        logger.error("Test initialization failed.") 