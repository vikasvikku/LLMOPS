from langchain_huggingface import HuggingFaceEmbeddings
 
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Initializing our HuggingFace embedding model")

        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        logger.info("HuggingFace embedding model initialized successfully")
        
        return model
    except Exception as e:
        error_message =  CustomException("Error occured while loading embedding model", e)
        logger.error(str(error_message))
        raise error_message