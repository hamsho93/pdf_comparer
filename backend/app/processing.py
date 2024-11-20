import os
import tempfile
from flask import jsonify
from app.utils import (
    encode_pdf_hierarchical_with_custom_chunking,
    pair_chunks_by_section_and_similarity,
    semantic_change_detection
)
from langchain_openai import OpenAIEmbeddings
import logging
from fastapi import UploadFile

# Load OpenAI embeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)

async def process_uploaded_files(file1: UploadFile, file2: UploadFile):
    """
    Process the uploaded files, compare them semantically, and generate GitHub-style diffs.
    """
    logging.debug("Starting file processing")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file2:
        temp_file1.write(await file1.read())
        temp_file2.write(await file2.read())
        temp_path1 = temp_file1.name
        temp_path2 = temp_file2.name

    try:
        logging.debug("Encoding PDFs")
        old_detailed_chunks = await encode_pdf_hierarchical_with_custom_chunking(temp_path1)
        new_detailed_chunks = await encode_pdf_hierarchical_with_custom_chunking(temp_path2)

        # Validate and filter out None values
        old_detailed_chunks = [chunk for chunk in old_detailed_chunks if chunk is not None]
        new_detailed_chunks = [chunk for chunk in new_detailed_chunks if chunk is not None]

        # logging.debug("Old detailed chunks: %s", old_detailed_chunks)
        # logging.debug("New detailed chunks: %s", new_detailed_chunks)

        if not old_detailed_chunks or not new_detailed_chunks:
            raise ValueError("Encoding function returned empty or invalid chunks")

        logging.debug("Pairing chunks")
        # Ensure you are iterating over the chunks correctly
        paired_chunks = await pair_chunks_by_section_and_similarity(
            old_chunks=old_detailed_chunks,
            new_chunks=new_detailed_chunks,
            embeddings=embeddings_model
        )

        logging.debug("Detecting semantic changes")
        changes = await semantic_change_detection(paired_chunks, embeddings_model)

        logging.debug("Generating diff results")
        diff_results = [
            {
                "type": change["type"],
                "old_text": change["old_text"],
                "new_text": change["new_text"],
                "similarity": change["similarity"],
                "diff": change["diff"],
                "metadata": {
                    "old_section": change["metadata"].get("old_section"),
                    "new_section": change["metadata"].get("new_section")
                }
            }
            for change in changes
        ]

        return diff_results

    except Exception as e:
        logging.error("Error in processing: %s", str(e))
        raise
    finally:
        if os.path.exists(temp_path1):
            os.remove(temp_path1)
        if os.path.exists(temp_path2):
            os.remove(temp_path2)
