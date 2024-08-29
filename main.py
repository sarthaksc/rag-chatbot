
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.document_loaders import WikipediaLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os



class ChatBot():

    def __init__(self):
        load_dotenv()
        wiki_loader = WikipediaLoader(query="Quantum mechanics", load_max_docs=1)
        wiki_doc = wiki_loader.load()
        recur_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=60,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            is_separator_regex=True,
        )
        data_splits = recur_splitter.split_documents(wiki_doc)
        embeddings = HuggingFaceEmbeddings()
        self.vectordb = FAISS.from_documents(data_splits, embeddings)
        self.retriever=self.vectordb.as_retriever()
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.8,
            top_p=0.8,
            top_k=50,
            huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_TOKEN")
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

        template = """
                  You are a Teacher. These students will ask you questions about quantum mechanics. Use the following piece of context to answer the question.
                  If you don't know the answer, just say you don't know.
                  Provide short and concise answers, no longer than 2 sentences.

                  Original Question: {question}
                  Context: {context}
                  Answer:
              """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        self.conversation_chain = ConversationalRetrievalChain.from_llm(llm=self.llm,
        retriever=self.retriever,
        memory=memory,
        return_source_documents=True)

    def ask(self,input):
        response=self.conversation_chain({"question": input})
        return response['answer']
