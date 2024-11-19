import asyncio
import os
from dotenv import load_dotenv
import cv2
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_utils import VectorDbWithBM25, LangchainLlms, RagFusion
from assistant_utils import DesktopScreenshot, EnhancedAssistant

async def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration
    pdf_directory = "rag_pdf"  # Directory containing LinkedIn-related documents
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("API key not found. Make sure your .env file is configured properly.")
    print(f"Loaded API Key: {openai_api_key[:5]}...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    # Initialize vector database
    vector_db = FAISS(
        embedding_function=embeddings,
        index=IndexFlatL2(1536),
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    # Load and process documents
    all_docs = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}")
            pdf_path = os.path.join(pdf_directory, filename)
            doc_loader = PyPDFLoader(pdf_path)
            pages = doc_loader.load_and_split()
            docs = text_splitter.split_documents(pages)
            all_docs.extend(docs)
    
    vector_db.add_documents(all_docs)
    
    # Initialize BM25 corpus
    bm25_corpus = [doc.page_content for doc in all_docs]
    
    #vectordbwithBM25 initialization failed, dm25 should be an input to initialization
    vector_db_with_bm25 = VectorDbWithBM25(vector_db = vector_db, bm25_corpus = bm25_corpus)
    langchain_llm = LangchainLlms()
    
    # try:
    #     print("Before RAG initialization...")
    #     rag = RagFusion(vector_store=vector_db_with_bm25, llm=langchain_llm.get_llm("OpenAI", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k").llm)
    #     print("After RAG initialization...")

    #     response = rag.arun("Provide LinkedIn message template suitable for reconnecting with an Anderson Alum Email Outreach for a Job, and fill in my name as Nei Fang and the subject name as Jack Smith", rewrite_original_query=False)
    #     print("RAG response:", response)
    # except Exception as e:
    #     print("An error occurred:", e)

    rag = RagFusion(vector_store=vector_db_with_bm25, llm=langchain_llm.get_llm("OpenAI", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k").llm)
    result = await rag.arun("Provide LinkedIn message template suitable for reconnecting with an Anderson Alum Email Outreach for a Job, and fill in my name as Nei Fang and the subject name as Jack Smith", rewrite_original_query=False)
    print("RAG response:", result)
    # # Initialize desktop screenshot and assistant
    # desktop_screenshot = DesktopScreenshot().start()
    # assistant = EnhancedAssistant(rag)

    # # Set up audio recognition
    # recognizer = Recognizer()
    # microphone = Microphone()
    # async def process_audio_input(prompt):
    # # Process prompt and screenshot input asynchronously
    #     screenshot = desktop_screenshot.read(encode=True)
    #     await assistant.answer(prompt, screenshot)
        
    # def audio_callback(recognizer, audio):
    #     try:
    #         prompt = recognizer.recognize_whisper(audio, model="base", language="english")
    #         assistant.answer(prompt, desktop_screenshot.read(encode=True))
    #     except UnknownValueError:
    #         print("There was an error processing the audio.")

    # # Main loop
    # try:
    #     while True:
    #         screenshot = desktop_screenshot.read()
    #         if screenshot is not None:
    #             cv2.imshow("Desktop", screenshot)
    #         if cv2.waitKey(1) in [27, ord("q")]:
    #             break
    # finally:
    #     desktop_screenshot.stop()
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())