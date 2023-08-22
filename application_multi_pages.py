#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.llms import OpenAI

# Dictionnaire pour stocker les paires de questions et réponses associées à chaque fichier PDF
qa_pairs_dict = {}
current_pdf = None

def process_pdf_with_openai(pdf_path, openai_code, question):
    os.environ["OPENAI_API_KEY"] = openai_code
    
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        raw_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            raw_text += page.extract_text()
        print(raw_text)
    # Split the text into smaller chunks for efficient processing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    print("Downloading embeddings...")
    embeddings = OpenAIEmbeddings()
    print("Embeddings downloaded successfully.")

    # Create the FAISS vector store for text documents
    docsearch = FAISS.from_texts(texts, embeddings)

    # Load the question-answering chain
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(llm=OpenAI(temperature=0.1))

    # Define the query
    query = question

    # Perform similarity search and retrieve relevant document
    docs = docsearch.similarity_search(query)

    # Run the question-answering chain on the input documents and query
    answers = chain.run(input_documents=docs, question=query)

    return answers


def main():
    current_pdf = st.session_state.get("current_pdf")
    if current_pdf is None:
        current_pdf = ""
    st.title("Interface interactive pour traiter un PDF avec OpenAI")

    # Page 1 : Demande d'entrer le code OpenAI
    with st.sidebar:
        st.title("Communiquer avec votre PDF")
        openai_code = st.text_input("Entrez votre code OpenAI :")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Pages", ["Communiquer avec votre PDF", "Télécharger le PDF", "Questions et réponses"])

    if page == "Communiquer avec votre PDF":
        # Page 1 : Demande d'entrer le code OpenAI
        if openai_code:
            st.write("Votre code OpenAI a été saisi.")
            st.subheader("Télécharger le PDF")
            st.write("Veuillez passer à la page suivante pour télécharger le PDF.")
        else:
            st.warning("Veuillez saisir votre code OpenAI.")

    elif page == "Télécharger le PDF":
        # Page 2 : Demande d'entrer le PDF
        with st.sidebar:
            pdf_file = st.file_uploader("Télécharger le fichier PDF", type=["pdf"])

        if pdf_file:
            current_pdf = pdf_file.name
            if current_pdf not in qa_pairs_dict:
                qa_pairs_dict[current_pdf] = []
            st.write("Votre fichier PDF a été téléchargé.")
            st.subheader("Questions et réponses")
            st.write("Veuillez passer à la page suivante pour saisir vos questions et obtenir les réponses.")

    elif page == "Questions et réponses":
        # Page 3 : Afficher les questions précédentes et leurs réponses pour le fichier PDF actuel
        st.title("Questions précédentes et leurs réponses pour le fichier PDF actuel")
        if current_pdf:
            if current_pdf in qa_pairs_dict:
                for q, a in qa_pairs_dict[current_pdf]:
                    st.write("Question:", q)
                    st.write("Réponse:", a)
        # Saisie de la nouvelle question
        question = st.text_input("Saisissez votre question :", key="question_input")
        if st.button("Obtenir la réponse"):
            if question:
                if current_pdf:
                    pdf_path = "temp.pdf"
                    # ... (votre code de traitement ici)
                    # Ajouter la paire de question et réponse au dictionnaire
                    qa_pairs_dict.setdefault(current_pdf, []).append((question, response))
                else:
                    st.warning("Veuillez télécharger un fichier PDF avant de poser une question.")
            else:
                st.warning("Veuillez saisir une question.")
if __name__ == "__main__":
    main()


# In[ ]:




