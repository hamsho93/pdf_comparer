from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .processing import process_uploaded_files
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "PDF Comparison API"}

@app.post("/compare")
async def compare_files(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    logger.debug("Received files for comparison")
    try:
        result = await process_uploaded_files(file1, file2)
        return result
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}", exc_info=True)
        raise