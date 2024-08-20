from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# loader = UnstructuredFileLoader("ai_adoption_framework_whitepaper.pdf")
loader = WebBaseLoader(
    "https://github.com/bitfumes/Langchain-RAG-system-with-Llama3-and-ChromaDB/blob/de393579e2de8b96620704a56250abe1cd99f97d/ai_adoption_framework_whitepaper.pdf?raw=true"
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)


# text_splitter = CharacterTextSplitter(
#     separator="\n", chunk_size=2000, chunk_overlap=200
# )
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(texts, embeddings)

llm = Ollama(
    model="uncensored-phi3",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
)

question = "Can you please summarize the document"
result = chain.invoke({"query": question})

print(result["result"])
