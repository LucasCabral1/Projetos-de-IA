from streamlit as st
import PyPDF2
import io
import os
import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Resume Critiquer", page_icon="📄", layout="centered")

st.title("AI Resume Analiser")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

def main():
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)


if __name__ == "__main__":
    main()
