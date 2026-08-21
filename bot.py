# import ollama
# import streamlit as st

# st.title("My First Ragbot ")
# with open("prompteng.txt","r") as file:
#     text = file.read()
# chunks = text.split("\n\n")#input is chunk which nedds to be converted into vector 
# response = ollama.embed(
#         model = "nomic-embed-text",
#         input = chunks 
# )
# vector = response["embeddings"]
# print(vector)

import ollama
import streamlit as st
import numpy as np 
st.title("My First Ragbot ")
with open("prompteng.txt","r") as file:
    text = file.read()
chunks = text.split("\n\n")
chunk_vectors = []  # empty list 
for chunk in chunks : 
    response = ollama.embed(
        model = "nomic-embed-text",
        input = chunk #to get chunk one by one  
    )
    vector = response["embeddings"][0]

    chunk_vectors.append(vector) # append adds vector in chunk_vector 

#questions 
question = st.chat_input("Ask Something..")
if question:
    with st.chat_message("user"):
        st.write(question)
    response = ollama.embed(
        model = "nomic-embed-text",
        input = question 
    )
    question_vector = response["embeddings"][0]

    #similarity search
    score = [] 
    for vector in chunk_vectors :
        similarity = np.dot(question_vector,vector)/( # FORMULA 
            np.linalg.norm(question_vector)* # LINALG = USED TO FIND LENGTH 
            np.linalg.norm(vector)
        )
        score.append(similarity)

    #finding best chunk 
    best_index = np.argmax(score) #max value 
    best_chunk = chunks[best_index]

    #create prompt
    prompt = f"""Answer the question based on the context below.

    Context:
    {best_chunk}

    Question: {question}"""


    #ask ollama
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]
    #display answer
    with st.chat_message("assistant"):
        st.write(answer)