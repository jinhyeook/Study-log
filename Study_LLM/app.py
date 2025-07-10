# UI 적용
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import  ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

load_dotenv()
api_key = os.getenv("open_api_key")

@st.cache_resource
def proces_pdf():
    url = r"C:\Users\kimji\Desktop\ProgramFile\Study\2024 KB 부동산 보고서_최종.pdf"
    loader = PyPDFLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    return  text_splitter.split_documents(documents)
    
@st.cache_resource
def init_vectorstore():
    chunks = proces_pdf()
    embdiing_function = OpenAIEmbeddings(api_key=api_key)
    return Chroma.from_documents( documents = chunks, embedding=embdiing_function,)
    
@st.cache_resource
def init_chain():
    vector_store = init_vectorstore()
    
    # 1) 검색 및 재정렬
    retriever = vector_store.as_retriever(search_kwards={"k":3})

    # 2) 프롬프트 템플릿 설정
    template = """ 당신은 KB 부동산 보고서 전문가 입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요. 컨텍스트 ; {context} """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("placeholder", "{chat_history}"), # 대화 기록용, 이전 대화 내용을 삽입.
            ("human", "{question}")
        ]
    ) 
    
    model = ChatOpenAI(model_name = "gpt-4o-mini", temperature=0, api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context = lambda x : format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        lambda session_id : ChatMessageHistory(),
        input_messages_key="question",
        history_messages_key = "chat_history",
    )


def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇")
    st.title("KB 부동산 보고서 AI 어드바이서")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    if ("messages" not in st.session_state):
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("부동산 관련 질문을 입력하세요") : 
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user", "content":prompt})

        chain = init_chain()

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question" : prompt},
                    {"configurable" : {"session_id" : "streamlit_session"}}
                )    
                st.markdown(response)

if __name__ == "__main__":
    main()   
