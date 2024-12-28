from openai import OpenAI
import psycopg2
import streamlit as st
from psycopg2.extensions import register_adapter, AsIs
import os
from dotenv import load_dotenv
load_dotenv()


api_key  = os.getenv("OPENAI_API_KEY")
openai = OpenAI()


#postgre connection

conn = psycopg2.connect(
    dbname = os.getenv("POSTGRESS_DB_NAME"),
    user = os.getenv("POSTGRESS_DB_USER"),
    password = os.getenv("POSTGRESS_DB_PASSWORD"),
    host = os.getenv("POSTGRESS_DB_HOST"),
    port = os.getenv("POSTGRESS_DB_PORT")

)

cursor = conn.cursor()


# simple chunks

documents = [
    "What is machine learning? Machine learning (ML) is a branch of artificial intelligence (AI) focused on enabling computers and machines to imitate the way that humans learn, to perform tasks autonomously, and to improve their performance and accuracy through experience and exposure to more data.",
    "UC Berkeley breaks out the learning system of a machine learning algorithm into three main parts.",

    "1. A Decision Process: Machine learning algorithms are used to make a prediction or classification. Based on some input data, which can be labeled or unlabeled, your algorithm will produce an estimate about a pattern in the data.",
    "2. An Error Function: An error function evaluates the prediction of the model. If there are known examples, an error function can make a comparison to assess the accuracy of the model.",

    "3. A Model Optimization Process: If the model can fit better to the data points in the training set, then weights are adjusted to reduce the discrepancy between the known example and the model estimate.",
    
    "The algorithm will repeat this iterative 'evaluate and optimize' process, updating weights autonomously until a threshold of accuracy has been met."
]



st.title("Interactive Q&A system")
st.write("This application demonestarte the steps in a Q&A")


# embeddings
for doc in documents:
    response = openai.embeddings.create(
        input = doc,
        model = "text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    cursor.execute(
        "INSERT INTO document_chunks (content, embedding) VALUES (%s, %s)",
        (doc, embedding)
    )

st.write("successfully store the embeddings")

  
conn.commit()


question = st.text_input("Enter your question")

def get_relavent_chunks(question, top_n=3):
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
    
if question:
    relavent_chunks = get_relavent_chunks(question)
    st.write("Top relevent chunks")
    for i, chunks in enumerate(relavent_chunks, start=1):
        st.write(f"{i}.{chunks}")


cursor.close()
conn.close()





# user_input = st.text_input("enter you tesxt")

# response = openai.chat.completions.create(
#             model="gpt-4o-mini-2024-07-18",
#             messages=[{"role": "user", "content": user_input}]
#         )

# result = response.choices[0].message
# st.write(result.content)
