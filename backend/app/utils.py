import asyncio
import random
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from difflib import unified_diff
import logging
import sys
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from openai import AsyncOpenAI
from asyncio import Semaphore
import backoff  # You'll need to install this: pip install backoff
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Add these filters to reduce noise
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key="your-api-key")
embeddings = client.embeddings

# Constants
EMBEDDING_TIMEOUT = 30  # Increased timeout
MAX_CONCURRENT_REQUESTS = 2  # Reduced concurrency
BATCH_SIZE = 3  # Smaller batch size
BATCH_DELAY = 1.0  # Longer delay between batches
MAX_RETRIES = 3  # Fewer retries but with longer timeouts

embedding_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [t for t in self.requests if now - t < 60]
            
            if len(self.requests) >= self.requests_per_minute:
                # Wait until we can make another request
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.requests.append(now)

rate_limiter = RateLimiter(requests_per_minute=40)  # Conservative rate limit

def should_retry(e):
    """Determine if we should retry based on the error"""
    return isinstance(e, (TimeoutError, asyncio.TimeoutError)) or 'rate limit' in str(e).lower()

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_RETRIES,
    max_time=60,  # Maximum total time to try
    giveup=lambda e: not should_retry(e)
)
async def get_embedding_with_retry(embeddings, text, timeout=EMBEDDING_TIMEOUT):
    """Get embeddings with improved rate limiting and retry logic"""
    start_time = time.time()
    
    try:
        # Acquire rate limit token
        await rate_limiter.acquire()
        
        async with embedding_semaphore:
            # Truncate long texts
            if len(text) > 8000:
                logger.warning(f"Text too long ({len(text)} chars), truncating...")
                text = text[:8000]
            
            # Add jitter to timeout to prevent thundering herd
            timeout_with_jitter = timeout + random.uniform(0, 5)
            
            embedding = await asyncio.wait_for(
                asyncio.to_thread(embeddings.embed_query, text),
                timeout=timeout_with_jitter
            )
            
            duration = time.time() - start_time
            if duration > 5:  # Increased threshold for slow request warning
                logger.warning(f"Slow embedding request: {duration:.2f}s")
            
            return embedding
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Embedding failed after {duration:.2f}s: {str(e)}")
        raise

async def process_chunk_batch(batch, embeddings):
    """Process a batch of chunks with better error handling"""
    results = []
    for chunk in batch:
        try:
            embedding = await get_embedding_with_retry(embeddings, chunk.page_content)
            results.append((chunk, embedding))
        except Exception as e:
            logger.error(f"Failed to process chunk: {str(e)}")
            results.append((chunk, None))
        # Add delay between individual requests in a batch
        await asyncio.sleep(0.5)
    return results

async def pair_chunks_by_section_and_similarity(old_chunks, new_chunks, embeddings, similarity_threshold=0.8):
    """Improved chunk pairing with batching"""
    logger.info(f"Starting pairing with {len(old_chunks)} old chunks and {len(new_chunks)} new chunks")
    
    try:
        paired_chunks = []
        used_new_chunks = set()
        
        # Process old chunks in small batches
        for i in range(0, len(old_chunks), BATCH_SIZE):
            batch = old_chunks[i:i + BATCH_SIZE]
            old_results = await process_chunk_batch(batch, embeddings)
            
            # Add delay between batches
            await asyncio.sleep(BATCH_DELAY)
            
            for old_chunk, old_embedding in old_results:
                if old_embedding is None:
                    continue
                    
                best_match = None
                best_similarity = 0.0
                
                # Process new chunks for each old chunk
                for new_chunk in new_chunks:
                    if new_chunks.index(new_chunk) in used_new_chunks:
                        continue
                        
                    try:
                        new_embedding = await get_embedding_with_retry(embeddings, new_chunk.page_content)
                        if new_embedding:
                            similarity = cosine_similarity([old_embedding], [new_embedding])[0][0]
                            if similarity > best_similarity and similarity >= similarity_threshold:
                                best_match = new_chunk
                                best_similarity = similarity
                    except Exception as e:
                        logger.error(f"Error processing new chunk: {str(e)}")
                        continue
                
                if best_match:
                    paired_chunks.append((old_chunk, best_match))
                    used_new_chunks.add(new_chunks.index(best_match))
                else:
                    paired_chunks.append((old_chunk, None))
        
        # Add remaining new chunks
        for i, new_chunk in enumerate(new_chunks):
            if i not in used_new_chunks:
                paired_chunks.append((None, new_chunk))
        
        return paired_chunks
        
    except Exception as e:
        logger.error(f"Error in chunk pairing: {str(e)}", exc_info=True)
        raise


def normalize_text(text):
    """
    Normalize text by removing extra spaces and standardizing line breaks.
    """
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def get_text_diff(old_text, new_text):
    """
    Generate GitHub-style diff using difflib.
    """
    old_text = normalize_text(old_text)
    new_text = normalize_text(new_text)
    diff = unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        lineterm=""
    )
    return "\n".join(diff)


def classify_changes(old_text, new_text, similarity, threshold=0.8):
    """
    Classify changes between old and new text.
    """
    if similarity >= threshold:
        if old_text == new_text:
            return "Exact Match"
        else:
            return "Rephrased"
    else:
        return "Substantial Change"


def extract_section(text):
    """Extract section information from text content"""
    # Common section headers in insurance documents
    section_patterns = [
        r"DECLARATIONS PAGE",
        r"SCHEDULED JEWELRY",
        r"POLICY DETAILS",
        r"ENDORSEMENTS",
        r"POLICY FEES",
        r"FORM NUMBER",
        r"WITNESS",
        r"PERSONAL JEWELRY POLICY"
    ]
    
    # Try to find a matching section header
    for pattern in section_patterns:
        if re.search(pattern, text.upper()):
            return pattern.title()
    
    # Try to extract section from page headers
    page_header = text.split('\n')[0] if text else ''
    if page_header and len(page_header.strip()) < 100:  # Reasonable header length
        return page_header.strip()
    
    return None

async def encode_pdf_hierarchical_with_custom_chunking(path, chunk_size=1000, chunk_overlap=200):
    """Encodes a PDF with section metadata"""
    try:
        loader = PyPDFLoader(path)
        documents = await asyncio.to_thread(loader.load)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
        
        # Add section metadata to chunks
        processed_chunks = []
        for chunk in detailed_chunks:
            # Extract section from chunk content
            section = extract_section(chunk.page_content)
            
            # Update metadata
            chunk.metadata.update({
                "section": section,
                "chunk_id": len(processed_chunks),
                "summary": False
            })
            processed_chunks.append(chunk)
            
            logger.debug(f"Processed chunk with section: {section}")
        
        return processed_chunks

    except Exception as e:
        logger.error(f"Error in PDF encoding: {str(e)}", exc_info=True)
        raise

async def semantic_change_detection(paired_chunks, embeddings, threshold=0.8):
    """Perform semantic change detection with section metadata"""
    try:
        changes = []
        for pair in paired_chunks:
            old_chunk, new_chunk = pair
            
            if old_chunk is None and new_chunk is not None:
                changes.append({
                    "type": "New Content",
                    "old_text": None,
                    "new_text": new_chunk.page_content,
                    "similarity": 0.0,
                    "diff": "This content is new.",
                    "metadata": {
                        "new_section": new_chunk.metadata.get("section"),
                        "old_section": None,
                        "new_page": new_chunk.metadata.get("page"),
                        "old_page": None
                    }
                })
                continue

            if new_chunk is None and old_chunk is not None:
                changes.append({
                    "type": "Removed Content",
                    "old_text": old_chunk.page_content,
                    "new_text": None,
                    "similarity": 0.0,
                    "diff": "This content was removed.",
                    "metadata": {
                        "old_section": old_chunk.metadata.get("section"),
                        "new_section": None,
                        "old_page": old_chunk.metadata.get("page"),
                        "new_page": None
                    }
                })
                continue

            # Both chunks exist
            old_embedding = await get_embedding_with_retry(embeddings, old_chunk.page_content)
            new_embedding = await get_embedding_with_retry(embeddings, new_chunk.page_content)
            
            similarity = cosine_similarity([old_embedding], [new_embedding])[0][0]
            change_type = classify_changes(old_chunk.page_content, new_chunk.page_content, similarity, threshold)
            diff = get_text_diff(old_chunk.page_content, new_chunk.page_content)

            changes.append({
                "type": change_type,
                "old_text": old_chunk.page_content,
                "new_text": new_chunk.page_content,
                "similarity": similarity,
                "diff": diff,
                "metadata": {
                    "old_section": old_chunk.metadata.get("section"),
                    "new_section": new_chunk.metadata.get("section"),
                    "old_page": old_chunk.metadata.get("page"),
                    "new_page": new_chunk.metadata.get("page")
                }
            })

        return changes

    except Exception as e:
        logger.error(f"Error in change detection: {str(e)}", exc_info=True)
        raise