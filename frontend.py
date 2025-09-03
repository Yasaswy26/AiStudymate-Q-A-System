# frontend.py
import streamlit as st
import requests

st.title("StudyMate — PDF Q&A")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    res = requests.post("http://127.0.0.1:8000/upload", files={"file": uploaded_file})
    if res.ok:
        st.success(f"Uploaded {uploaded_file.name} successfully!")
    else:
        st.error(f"Upload failed: {res.text}")

# Ask question
st.subheader("Ask Questions from PDFs")
question = st.text_input("Your question:")
k = st.slider("Number of retrieved chunks (k)", 1, 10, 5)

if st.button("Get Answer") and question:
    data = {"question": question, "k": k}
    response = requests.post("http://127.0.0.1:8000/ask", json=data)
    if response.ok:
        st.success(response.json().get("answer", "No answer returned"))
    else:
        st.error(f"Error: {response.text}")
