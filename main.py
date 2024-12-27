import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

!pip install transformers torch indobenchmark

# Load the BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Summarisation function
def summarise_text(text, max_length=130, min_length=30, do_sample=False):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True, do_sample=do_sample)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Keyword extraction function
def extract_keywords_tfidf(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]
    keywords = [(feature_names[i], scores[i]) for i in top_indices]
    return keywords

# Streamlit UI
st.title("AI Interview Analysis Tool")

# Upload video file
uploaded_file = st.file_uploader("Upload Interview Video", type=["mp4"])
if uploaded_file:
    # Save video file
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Extract audio and transcribe
    try:
        video = mp.VideoFileClip("uploaded_video.mp4")
        audio_file = video.audio
        audio_file.write_audiofile("interview_audio.wav")

        language_choice = st.selectbox("Choose transcription language", ['English (en)', 'Indonesian (id)'])
        language_code = "en-US" if "English" in language_choice else "id-ID"
        
        r = sr.Recognizer()
        with sr.AudioFile("interview_audio.wav") as source:
            data = r.record(source)
        text = r.recognize_google(data, language=language_code)
        
        st.success("Transcription complete!")
        st.text_area("Transcribed Text", value=text, height=200)

        # Summarise the transcription
        summary = summarise_text(text)
        st.subheader("Summary")
        st.write(summary)

        # Extract keywords
        keywords = extract_keywords_tfidf(text, top_n=5)
        keyword_list = [keyword for keyword, score in keywords]
        st.subheader("Keywords")
        st.write(keyword_list)

        # Save results to CSV
        results = {
            'Transcription': text,
            'Summarisation': summary,
            'Keywords': ", ".join(keyword_list)
        }
        df = pd.DataFrame([results])
        csv_file_name = 'analysis_results.csv'
        df.to_csv(csv_file_name, index=False)
        st.success(f"Results saved to {csv_file_name}")

        # Provide download link
        with open(csv_file_name, "rb") as file:
            st.download_button(label="Download Results", data=file, file_name=csv_file_name, mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
