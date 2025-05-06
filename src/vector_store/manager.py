"""
Manages the vector database (ChromaDB) interactions for storing and retrieving
transcript chunks and their embeddings.
"""

import logging
import time
import nltk
import chromadb
from chromadb.utils import embedding_functions # Use newer import style if needed based on ChromaDB version
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
VECTOR_DB_PATH = "vector_db"  # Directory to store the persistent ChromaDB database
COLLECTION_NAME = "transcripts" # Name of the collection within ChromaDB
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # A popular, efficient sentence transformer
# Alternative: Use ChromaDB's built-in SentenceTransformer embedding function
# This might simplify setup if sentence-transformers library has issues
# If using this, you might not need the explicit SentenceTransformer import/init later
DEFAULT_EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# --- Initialization ---
_chroma_client = None
_embedding_function = None # Store the function or model instance globally within the module

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
    # Using ChromaDB's built-in wrapper for simplicity
    global _embedding_function
    if _embedding_function is None:
        logger.info(f"Using ChromaDB SentenceTransformer embedding function ({DEFAULT_EF.model_name}).")
        _embedding_function = DEFAULT_EF
        # Add error handling if needed
    return _embedding_function

def initialize_vector_store(collection_name: str = COLLECTION_NAME):
    """
    Initializes the vector store: ensures NLTK data is downloaded,
    connects to ChromaDB, and gets or creates the specified collection.

    Returns:
        The ChromaDB collection object, or None if initialization fails.
    """
    logger.info(f"Initializing vector store collection: '{collection_name}'...")
    try:
        # 1. Ensure NLTK sentence tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            logger.info("Downloading NLTK 'punkt' tokenizer...")
            nltk.download('punkt', quiet=True)
            logger.info("'punkt' tokenizer downloaded.")
        except Exception as nltk_e:
             logger.warning(f"Could not find or download NLTK 'punkt': {nltk_e}. Sentence tokenization might fail.")
             # Decide if this is fatal or can proceed

        # 2. Get ChromaDB client
        client = get_chroma_client()
        if not client:
            return None # Error logged in get_chroma_client

        # 3. Get or create the collection with the embedding function
        embedding_func = get_embedding_function()
        if not embedding_func:
             logger.error("Failed to get embedding function.")
             return None

        logger.info(f"Getting or creating ChromaDB collection '{collection_name}'...")
        # Pass embedding_function directly if using ChromaDB's wrapper >= 0.4.0
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_func, # Use the embedding function instance
            metadata={"hnsw:space": "cosine"}  # Use cosine distance for sentence embeddings
        )
        logger.info(f"Vector store collection '{collection_name}' ready.")
        return collection

    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
        return None

# --- Store Transcript Implementation ---
def store_transcript(transcript_text: str, source_id: str, collection):
    """
    Chunks the transcript, generates embeddings (implicitly via collection), 
    and stores them in the collection.
    """
    if not transcript_text or not source_id or not collection:
        logger.error("store_transcript called with invalid arguments.")
        return False
        
    logger.info(f"Storing transcript for source_id: '{source_id}'...")
    
    try:
        # 1. Chunk the transcript into sentences
        # Ensure NLTK data is available (redundant check, but safe)
        try:
            sentences = nltk.sent_tokenize(transcript_text)
        except LookupError:
            logger.warning("NLTK 'punkt' tokenizer not found. Falling back to simple newline split.")
            # Fallback or re-download
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(transcript_text)
        except Exception as e:
             logger.error(f"Failed to tokenize transcript: {e}")
             return False # Cannot proceed without chunks

        if not sentences:
            logger.warning(f"Transcript for '{source_id}' resulted in zero sentences after tokenization.")
            return False
            
        logger.info(f"Split transcript into {len(sentences)} sentence chunks.")
        
        # TODO: Optional - Implement more sophisticated chunking (e.g., sliding windows, size limits)
        # For now, using sentences as chunks.
        chunks = sentences
        
        # 2. Create IDs and Metadata
        ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        metadata = [{'source': source_id, 'chunk_index': i, 'timestamp': time.time()} for i in range(len(chunks))]
        
        # 3. Add to ChromaDB Collection
        # The embedding generation happens automatically here if an embedding_function was provided
        # when the collection was created/retrieved.
        logger.info(f"Adding {len(chunks)} chunks to collection '{collection.name}'...")
        collection.add(
            embeddings=None, # Let the collection's embedding function handle this
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
        logger.info(f"Successfully stored transcript chunks for '{source_id}'.")
        return True

    except Exception as e:
        logger.error(f"Failed to store transcript for '{source_id}': {e}", exc_info=True)
        return False

# --- Search Transcripts Implementation ---
def search_transcripts(query: str, collection, n_results: int = 5) -> list[str]:
    """
    Embeds the query (implicitly via collection) and searches the collection 
    for relevant document chunks.
    
    Args:
        query: The user's search query string.
        collection: The initialized ChromaDB collection object.
        n_results: The maximum number of relevant chunks to return.
        
    Returns:
        A list of the relevant document text chunks (strings).
    """
    if not query or not collection:
        logger.error("search_transcripts called with invalid arguments.")
        return []
        
    logger.info(f"Searching collection '{collection.name}' for query: '{query:.50}...' (n_results={n_results})")
    
    try:
        # The query embedding also happens automatically via the collection's embedding function
        results = collection.query(
            query_texts=[query], # Pass the query text directly
            n_results=n_results,
            include=['documents'] # Only need the document text for now
        )
        
        # Extract the documents from the results
        # Results structure is typically like: {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'embeddings': None, 'documents': [[doc1, doc2,...]]}
        retrieved_docs = results.get('documents', [[]])[0]
        logger.info(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        return retrieved_docs
        
    except Exception as e:
        logger.error(f"Failed to search transcripts: {e}", exc_info=True)
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