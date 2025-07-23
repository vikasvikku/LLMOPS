import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path doesnt exist")
        
        logger.info(f"loading files from {DATA_PATH}")
        import glob
        pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        
        if not documents:
            logger.warning("No pdfs were found")
        else:
            logger.info(f"Successfully fetched {len(documents)} documents")

        return documents
    
    except Exception as e:
        logger.error(f"Failed to load pdf: {e}")
        return[]

def create_text_chunk(documents):
    try:
        if not documents:
            raise CustomException("No documents were found")
        
        logger.info(f"splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)}  text chunks")

        return text_chunks
    
    except Exception as e:
        logger.error(f"Failed to generate chunks: {e}")
        return[]

