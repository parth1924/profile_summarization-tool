import PyPDF2
import nltk
import google.generativeai as genai
import streamlit as st
from io import BytesIO
import json
import mysql.connector
from mysql.connector import Error
import re

# Configure the Google Gemini API
genai.configure(api_key=st.secrets["general"]["api_key"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL Database: {e}")
        return None

def parse_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_words)

def extract_json_from_markdown(text):
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    return json_match.group(1) if json_match else text

def llm_process(text):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", generation_config=generation_config
    )
    chat_session = model.start_chat(history=[])
    prompt = (
        f"Analyze this text: {text}\n"
        f"Provide the following information in JSON format:\n"
        f"1. 'name': The full name of the person\n"
        f"2. 'email': The email address of the person\n"
        f"3. 'summary': A 3-sentence summary of the person's profile\n"
        f"4. 'top_skills': An array of the top 3 fields of expertise based on the profile\n"
        f"Ensure that the JSON keys are exactly as specified: 'name', 'email', 'summary', and 'top_skills'."
    )
    response = chat_session.send_message(prompt)
    return response.text.strip()

def email_exists(connection, email):
    try:
        cursor = connection.cursor()
        query = "SELECT email FROM profiles WHERE email = %s"
        cursor.execute(query, (email,))
        result = cursor.fetchone()
        return result is not None
    except Error as e:
        st.error(f"Error checking email in MySQL Database: {e}")
        return False
    finally:
        if cursor:
            cursor.close()

def insert_profile(connection, profile_data):
    if email_exists(connection, profile_data.get('email', '')):
        st.warning("Profile with this email already exists in the database.")
        return
    
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO profiles (name, email, summary, top_fields)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (
            profile_data.get('name', ''),
            profile_data.get('email', ''),
            profile_data.get('summary', ''),
            json.dumps(profile_data.get('top_skills') or profile_data.get('top_fields', []))
        ))
        connection.commit()
        st.success("Profile data inserted successfully!")
    except Error as e:
        st.error(f"Error inserting data into MySQL Database: {e}")
    finally:
        if cursor:
            cursor.close()

def summarize_and_extract(pdf_file):
    text = parse_pdf(pdf_file)
    processed_text = preprocess_text(text)
    response = llm_process(processed_text)
    return response

# Streamlit app UI
st.title("PDF Summarizer and Expertise Extractor")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        response = summarize_and_extract(uploaded_file)
        json_content = extract_json_from_markdown(response)

        try:
            profile_data = json.loads(json_content)
            st.write("Extracted Profile Data:")
            st.json(profile_data)

            connection = create_db_connection()
            if connection:
                insert_profile(connection, profile_data)
                connection.close()
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response from LLM: {e}")
            st.write("Raw JSON content:")
            st.code(json_content, language='json')
