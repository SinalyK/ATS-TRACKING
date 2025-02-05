from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(groq_api_key=groq_api_key,model="Gemma2-9b-It")


#Streamlit App
st.title("ATS Tracking System")
input_text=st.text_area("Job Description")
upload_file=st.file_uploader("Upload your CV(PDF)",type=['pdf'],accept_multiple_files=True)

##Document Loading
if upload_file is not None:
    for file in upload_file:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(file.getvalue())
            file_name=file.name

        docs=PyPDFLoader(temppdf)
        st.success("PDF Uploaded Succesfully Choose one option")
        docs=docs.load()
        text_splitters=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=70)
        splits=text_splitters.split_documents(docs)
        vectorStore=FAISS.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorStore.as_retriever()


prompt1=ChatPromptTemplate.from_template(
"""
You are an expert Technical Human Resource Manager and you can understand many language,
your task is to review the provided resume against the job description for these profiles.
Please share your professional evaluation on wether the candidate's profile aligns with Highlight 
the strenghs and weakness of the applicant in relation to the specified job requirements and list the missing keywords in the resume language 
<context>
{context}
</context>
Job description:{input}
"""
)

prompt2=ChatPromptTemplate.from_template(
"""
you are an skilled ATS(Applicant Tracking System) scanner with a deep understand in ATS functionality and you can understand many language 
so give the response based in {lang}
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description.The output should come as percentage
{context}
</context>
Job description:{input}
"""
)

#btn1=st.button("Tell me about the CV")
btn2=st.button("what are the Keywords that are missied")
btn3=st.button("Percentage match")
lang=st.radio("Langue:",options=["English","French"])
if btn2 and upload_file is not None:
    stuff_chain=create_stuff_documents_chain(llm=llm,prompt=prompt1)
    chain=create_retrieval_chain(retriever,stuff_chain)
    response=chain.invoke({"input":input_text})
    st.write(response['answer'])
elif btn3 and upload_file is not None:
    stuff_chain=create_stuff_documents_chain(llm=llm,prompt=prompt2)
    chain=create_retrieval_chain(retriever,stuff_chain)
    response=chain.invoke({"lang":lang,"input":input_text})
    st.write(response['answer'])
    

