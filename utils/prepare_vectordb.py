import os
import shutil
import traceback
from typing import Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class PrepareVectorDB:
    def __init__(
        self,
        data_directory: Union[str, list],
        persist_directory: Union[str, os.PathLike],
        openai_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.file_path = data_directory[0] if isinstance(data_directory, list) else data_directory
        self.persist_directory = str(persist_directory)
        self.openai_api_key = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _load_document(self):
        ext = self.file_path.split(".")[-1].lower()

        loader_map = {
            "pdf": PyPDFLoader,
            "docx": UnstructuredWordDocumentLoader,
            "pptx": UnstructuredPowerPointLoader,
            "xlsx": UnstructuredExcelLoader,
            "txt": TextLoader,
        }

        if ext not in loader_map:
            raise ValueError(f"❌ Unsupported file format for RAG: .{ext}")

        loader_cls = loader_map[ext]
        print(f"📄 Loading document using {loader_cls.__name__}: {self.file_path}")

        try:
            loader = loader_cls(str(self.file_path))
            documents = loader.load()
            if not documents or len(documents) == 0:
                raise ValueError(f"⚠️ No content extracted from {self.file_path}. Please check the file.")
            return documents
        except Exception as e:
            print(f"❌ Failed to load document: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

    def prepare_and_save_vectordb(self, attempt: int = 1):
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            print("📥 Starting vector DB preparation...")
            documents = self._load_document()
            print(f"✅ Document loaded: {len(documents)} chunk(s)")

            print("✂️ Splitting document into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            print(f"📚 Total chunks: {len(chunks)}")

            if not chunks:
                raise ValueError("❌ Document loaded but no text chunks were extracted.")

            print("🔍 Creating embeddings and initializing Chroma DB...")
            embedding_fn = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_fn,
                persist_directory=self.persist_directory
            )

            vectordb.persist()
            print(f"✅ Vector DB saved at: {self.persist_directory}")

        except Exception as e:
            print(f"❌ Failed to prepare vector store: {type(e).__name__}: {e}")
            traceback.print_exc()

            if attempt == 1 and ("no such column" in str(e).lower() or "tenant" in str(e).lower()):
                print("⚠️ Detected corrupt or incompatible vector DB. Deleting and retrying...")
                try:
                    shutil.rmtree(self.persist_directory, ignore_errors=True)
                    return self.prepare_and_save_vectordb(attempt=2)
                except Exception as cleanup_error:
                    print(f"❌ Failed to delete old DB: {type(cleanup_error).__name__}: {cleanup_error}")
                    traceback.print_exc()

            raise RuntimeError("💥 Vector store preparation failed. Try a different document or check your environment.") from e
