import os
import psycopg2
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("Interactive Q&A System")
st.write("This application demonstrates the steps in a Q&A")


api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI()
openai.api_key = api_key  

conn = psycopg2.connect(
    dbname = os.getenv("POSTGRESS_DB_NAME"),
    user = os.getenv("POSTGRESS_DB_USER"),
    password = os.getenv("POSTGRESS_DB_PASSWORD"),
    host = os.getenv("POSTGRESS_DB_HOST"),
    port = os.getenv("POSTGRESS_DB_PORT"))

cursor = conn.cursor()

documents = [
    "What is machine learning? Machine learning (ML) is a branch of artificial intelligence (AI) focused on enabling computers and machines to imitate the way that humans learn, to perform tasks autonomously, and to improve their performance and accuracy through experience and exposure to more data.",
    "UC Berkeley breaks out the learning system of a machine learning algorithm into three main parts.",
    "1. A Decision Process: Machine learning algorithms are used to make a prediction or classification. Based on some input data, which can be labeled or unlabeled, your algorithm will produce an estimate about a pattern in the data.",
    "2. An Error Function: An error function evaluates the prediction of the model. If there are known examples, an error function can make a comparison to assess the accuracy of the model.",
    "3. A Model Optimization Process: If the model can fit better to the data points in the training set, then weights are adjusted to reduce the discrepancy between the known example and the model estimate.",
    "The algorithm will repeat this iterative 'evaluate and optimize' process, updating weights autonomously until a threshold of accuracy has been met."
]


def document_exists(doc):
    cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE content = %s", (doc,))
    return cursor.fetchone()[0] > 0

for doc in documents:
    if not document_exists(doc):
        response = openai.embeddings.create(
            input=doc,
            model="text-embedding-3-small"  
        )
        
        embedding = response.data[0].embedding
        
        cursor.execute(
            "INSERT INTO document_chunks (content, embedding) VALUES (%s, %s)",
            (doc, embedding)
        )
        
conn.commit()


def get_relevant_chunks(question, top_n=3):
    question_embeddings = openai.embeddings.create(
        input = question,
        model = "text-embedding-3-small"
    )

    question_embeddings = question_embeddings.data[0].embedding

    cursor.execute(
        """
        SELECT content FROM document_chunks
        ORDER BY embedding <=> %s::vector  
        LIMIT %s
        """, (question_embeddings, top_n)
    )
    
    relavent_chunks = [row[0] for row in cursor.fetchall()]
    return relavent_chunks

question = st.text_input("Enter your question")

if question:
    relevant_chunks = get_relevant_chunks(question)
    
    context = "\n".join(f"{i + 1}. {chunk}" for i, chunk in enumerate(relevant_chunks))
    
    st.write(context)
    prompt = f"""
    Using the following information: 
    {context}

    Answer the question:
    {question}
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.1
    )
    
    result = response.choices[0].message
    st.write(result.content)

cursor.close()
conn.close()
