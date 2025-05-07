import streamlit as st
import pickle
import re
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity

# Load saved model files
df = pickle.load(open('df.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
matrix = pickle.load(open('matrix.pkl', 'rb'))

# Define multi-word job titles to preserve
multi_word_titles = ["web developer", "software engineer", "data scientist", "cloud engineer", "project intern"]

# NLP preprocessing (without stemming)
stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'about'])

def clean_text(text):
    # Replace multi-word job titles with a placeholder
    for title in multi_word_titles:
        text = text.replace(title, f"_{title.replace(' ', '_')}_")
    
    # Remove non-alphanumeric characters except spaces (preserve spaces)
    text = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text)
    
    # Tokenize the cleaned text using regex
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # Revert placeholders to original multi-word job titles
    cleaned_text = " ".join(cleaned_tokens)
    for title in multi_word_titles:
        cleaned_text = cleaned_text.replace(f"_{title.replace(' ', '_')}_", title)
    
    return cleaned_text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to recommend jobs based on the cleaned resume text
def recommend_jobs(resume_text):
    cleaned_resume = clean_text(resume_text)  # Clean text without stemming
    vector = tfidf.transform([cleaned_resume])
    sim_scores = cosine_similarity(vector, matrix)[0]
    top_indices = sim_scores.argsort()[-5:][::-1]
    
    # Refine the output and join words with commas
    refined_recommendations = df.iloc[top_indices][['Interests', 'Skills', 'JobTitle']].applymap(lambda x: ', '.join(x.split()))
    
    return refined_recommendations

# Streamlit app interface
st.title("📄 Resume-Based Job Recommendation System")
st.write("Upload your **resume (PDF or TXT)** and get personalized job suggestions!")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")
    
    st.subheader("📌 Recommended Jobs:")
    recommendations = recommend_jobs(resume_text)
    st.table(recommendations.reset_index(drop=True).rename(columns={
        'Interests': '💡 Interests',
        'Skills': '🛠️ Skills',
        'JobTitle': '🏷️ Job Title'
    }))
