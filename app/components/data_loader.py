import os
from app.components.pdf_loader import load_pdf_files,create_text_chunk
from app.components.vector_store import save_vector_store
from app.config.config import DB_FAISS_PATH

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def process_and_store_pdfs():
    try:
        logger.info("Making the Vectorstore.....")
        
        documents = load_pdf_files()

        text_chunks = create_text_chunk(documents)
        
        save_vector_store(text_chunks)

        logger.info("Vectorstore created successfully....")

    except Exception as e:
        error_message = CustomException("Failed to create vectorstore", e)
        logger.error(str(error_message))

    
if __name__ == "__main__":
    process_and_store_pdfs()
