import streamlit as st
from rake_nltk import Rake
import nltk
from langdetect import detect
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import string # For punctuation removal
from modules.nlp_engine import load_nltk_resources, load_spacy_model, load_sentiment_pipeline


# --- Configuration and Resource Loading ---
st.set_page_config(layout="wide", page_title="Advanced NLP Engine")
st.title("🧠 Advanced NLP Intelligence Engine")
st.markdown("Paste your text or upload a document to extract deep language insights using **AI and NLP**.")


# Load resources at the start
if load_nltk_resources():
    nlp_spacy = load_spacy_model()
    # Initialize the sentiment pipeline once
    classifier = load_sentiment_pipeline()
else:
    st.stop() # Stop if NLTK resources fail to load

# --- Input Section ---
st.markdown("---")
st.subheader("✍️ Input Your Text")

input_type = st.radio("Choose Input Type:", ["Text Input", "Upload File"], key="input_type_radio")

# Initialize session state for text area content
if "user_text_input" not in st.session_state:
    st.session_state.user_text_input = ""

text = ""
if input_type == "Text Input":
    text = st.text_area("Paste Text Below:", height=300, value=st.session_state.user_text_input, key="main_text_area")
else:
    uploaded_file = st.file_uploader("📄 Upload .txt or .pdf File (PDF support is basic, prefers TXT):", type=["txt", "pdf"], key="file_uploader")
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or "" # Extract text, handle None
                if not text.strip():
                    st.warning("Could not extract text from PDF. Please ensure it's text-searchable.")
            except ImportError:
                st.error("Please install `pypdf` to process PDF files: `pip install pypdf`")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
        else: # Assumes .txt
            text = uploaded_file.read().decode("utf-8")
        st.session_state.user_text_input = text # Update session state with uploaded text

# --- Preprocessing Options ---
st.markdown("---")
st.subheader("⚙️ Preprocessing Options")
col_pre1, col_pre2, col_pre3 = st.columns(3)
with col_pre1:
    do_lowercase = st.checkbox("Convert to Lowercase", value=True, key="preprocess_lowercase")
with col_pre2:
    remove_punctuation = st.checkbox("Remove Punctuation", value=True, key="preprocess_punctuation")
with col_pre3:
    remove_stopwords = st.checkbox("Remove Stopwords", value=True, key="preprocess_stopwords")

# --- Action Buttons ---
st.markdown("---")
col_buttons1, col_buttons2 = st.columns([1, 4])
with col_buttons1:
    analyze_button = st.button("🚀 Analyze Text", key="analyze_button")
with col_buttons2:
    if st.button("🧹 Clear Text Input", key="clear_text_button"):
        st.session_state.user_text_input = ""
        st.rerun() # Rerun to clear the text area and reset analysis


# --- Analysis Logic (triggered by button click) ---
if analyze_button:
    if not text.strip():
        st.info("Please enter or upload text to analyze.")
        st.stop() # Stop execution if no text is provided

    with st.spinner("Analyzing your text... This may take a moment."):
        processed_text = text

        if do_lowercase:
            processed_text = processed_text.lower()
        if remove_punctuation:
            processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize early for stopword removal and word frequency
        tokens = nltk.word_tokenize(processed_text)
        
        if remove_stopwords:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens = [w for w in tokens if w.lower() not in stop_words]
        
        # Rejoin tokens for analysis methods that expect a string
        final_text_for_analysis = " ".join(tokens)
        
        # --- Section: Language & Readability ---
        with st.expander("🌍 Language Detection & Readability", expanded=True):
            if not final_text_for_analysis.strip():
                st.warning("Not enough processed text for Language Detection & Readability analysis.")
            else:
                try:
                    lang = detect(final_text_for_analysis)
                    blob = TextBlob(final_text_for_analysis)
                    
                    st.markdown(f"- **Detected Language:** `{lang.upper()}`")
                    st.markdown(f"- **Polarity (Sentiment):** `{blob.sentiment.polarity:.3f}` (Range: -1.0 to 1.0, -1 is negative, 1 is positive)")
                    st.markdown(f"- **Subjectivity:** `{blob.sentiment.subjectivity:.3f}` (Range: 0.0 to 1.0, 0 is objective, 1 is subjective)")
                    st.markdown(f"- **Word Count:** `{len(tokens)}`")
                    st.markdown(f"- **Sentence Count:** `{len(blob.sentences)}`")
                except Exception as e:
                    st.error(f"Error in Language Detection & Readability: {e}")

        # --- Section: Sentiment Analysis (Transformer-based) ---
        with st.expander("❤️ Sentiment Analysis (Transformers)", expanded=False):
            if not final_text_for_analysis.strip():
                st.warning("Not enough processed text for Sentiment Analysis.")
            else:
                try:
                    # Transformers pipeline might have a max input length, typically 512 tokens.
                    # Truncate if necessary, but try to send as much as possible.
                    sentiment_result = classifier(final_text_for_analysis[:512])[0]
                    st.markdown(f"- **Label:** `{sentiment_result['label']}`")
                    st.markdown(f"- **Score:** `{sentiment_result['score']:.3f}`")
                except Exception as e:
                    st.error(f"Error in Sentiment Analysis (Transformers): {e}. Text might be too long or model issue.")

        # --- Section: Named Entity Recognition ---
        with st.expander("🧾 Named Entity Recognition (NER)", expanded=False):
            if nlp_spacy is None:
                st.warning("SpaCy model not loaded. Cannot perform NER.")
            elif not final_text_for_analysis.strip():
                st.warning("Not enough processed text for Named Entity Recognition.")
            else:
                try:
                    doc = nlp_spacy(final_text_for_analysis)
                    entities = [(ent.text, ent.label_) for ent in doc.ents]
                    if entities:
                        df_ents = pd.DataFrame(entities, columns=["Entity", "Label"])
                        st.dataframe(df_ents.drop_duplicates(), use_container_width=True)
                    else:
                        st.info("No named entities found.")
                except Exception as e:
                    st.error(f"Error in Named Entity Recognition: {e}")

        # --- Enhanced Section: Keyword Extraction ---
        with st.expander("🗝️ Keyword Extraction", expanded=False):
            if not final_text_for_analysis.strip():
                st.warning("Not enough processed text for Keyword Extraction.")
            else:
                st.markdown("#### Discover the most relevant terms and phrases in your text.")
                col_rake, col_tfidf = st.columns(2)

                # RAKE Keyword Extraction
                with col_rake:
                    st.markdown("##### 🔑 RAKE (Rapid Automatic Keyword Extraction)")
                    st.info("RAKE identifies keywords based on word frequency and co-occurrence within sentences.")
                    try:
                        # Rake instance with stopwords based on preprocessing choice
                        rake = Rake(stopwords=nltk.corpus.stopwords.words('english') if remove_stopwords else None)
                        rake.extract_keywords_from_text(final_text_for_analysis)
                        # Get phrases with scores
                        ranked_phrases_with_scores = rake.get_ranked_phrases_with_scores()[:10] # Top 10

                        if ranked_phrases_with_scores:
                            df_rake = pd.DataFrame(ranked_phrases_with_scores, columns=["Score", "Keyword/Phrase"])
                            df_rake["Score"] = df_rake["Score"].round(3) # Round scores
                            st.dataframe(df_rake, use_container_width=True, hide_index=True)

                            # Plot RAKE keywords
                            fig_rake, ax_rake = plt.subplots(figsize=(8, 5))
                            # Reverse for horizontal bar chart to have highest on top
                            words_rake = [item[1] for item in ranked_phrases_with_scores[::-1]]
                            scores_rake = [item[0] for item in ranked_phrases_with_scores[::-1]]
                            ax_rake.barh(words_rake, scores_rake, color='skyblue')
                            ax_rake.set_xlabel("Score")
                            ax_rake.set_title("Top RAKE Keywords")
                            plt.tight_layout()
                            st.pyplot(fig_rake)
                        else:
                            st.info("No keywords found by RAKE.")
                    except Exception as e:
                        st.error(f"Error in RAKE Keyword Extraction: {e}")

                # TF-IDF Keyword Extraction
                with col_tfidf:
                    st.markdown("##### 📈 TF-IDF (Term Frequency-Inverse Document Frequency)")
                    st.info("TF-IDF evaluates word importance by how often they appear in this document vs. other documents.")
                    try:
                        # TF-IDF needs at least 2 tokens to form a matrix, and a meaningful vocabulary
                        if len(set(tokens)) < 2: # Check unique tokens
                             st.warning("Not enough unique words for TF-IDF keyword extraction.")
                        else:
                            # Use `min_df=1` to ensure single-document TF-IDF works
                            tfidf_vectorizer = TfidfVectorizer(
                                stop_words='english' if remove_stopwords else None,
                                max_features=10, # Top 10 features
                                min_df=1 # Consider terms that appear at least once
                            )
                            # Fit and transform on the document
                            tfidf_matrix = tfidf_vectorizer.fit_transform([final_text_for_analysis])
                            feature_names = tfidf_vectorizer.get_feature_names_out()
                            
                            # Get scores for the first (and only) document
                            tfidf_scores = tfidf_matrix.toarray()[0]
                            
                            # Create DataFrame and sort by score
                            df_tfidf = pd.DataFrame({'Keyword': feature_names, 'Score': tfidf_scores})
                            df_tfidf = df_tfidf.sort_values(by='Score', ascending=False).head(10)
                            df_tfidf["Score"] = df_tfidf["Score"].round(3) # Round scores
                            
                            if not df_tfidf.empty:
                                st.dataframe(df_tfidf, use_container_width=True, hide_index=True)

                                # Plot TF-IDF keywords
                                fig_tfidf, ax_tfidf = plt.subplots(figsize=(8, 5))
                                # Reverse for horizontal bar chart to have highest on top
                                words_tfidf = df_tfidf['Keyword'].tolist()[::-1]
                                scores_tfidf = df_tfidf['Score'].tolist()[::-1]
                                ax_tfidf.barh(words_tfidf, scores_tfidf, color='lightcoral')
                                ax_tfidf.set_xlabel("Score")
                                ax_tfidf.set_title("Top TF-IDF Keywords")
                                plt.tight_layout()
                                st.pyplot(fig_tfidf)
                            else:
                                st.info("No keywords found by TF-IDF.")
                    except ValueError as ve:
                        st.warning(f"TF-IDF error: {ve}. This often happens with very short or homogeneous text.")
                    except Exception as e:
                        st.error(f"Error in TF-IDF Keyword Extraction: {e}")

        # --- Section: Summarization ---
        with st.expander("📝 Text Summarization", expanded=False):
            if not final_text_for_analysis.strip():
                st.warning("Not enough processed text for Summarization.")
            else:
                try:
                    # Sumy needs raw text, not tokenized, for its parser
                    parser = PlaintextParser.from_string(text, Tokenizer("english")) # Use original 'text'
                    summarizer = LsaSummarizer()
                    # Calculate sentences_count dynamically based on text length, min 1, max 5
                    # Ensuring we don't request more sentences than available
                    num_available_sentences = len(list(parser.document.sentences))
                    num_sentences_for_summary = max(1, min(5, num_available_sentences // 4)) 
                    num_sentences_for_summary = min(num_sentences_for_summary, num_available_sentences) # Ensure it doesn't exceed available
                    
                    if num_sentences_for_summary == 0 and num_available_sentences > 0: # If text is too short, try to get at least 1 sentence
                        num_sentences_for_summary = 1

                    if num_sentences_for_summary > 0:
                        summary = summarizer(parser.document, sentences_count=num_sentences_for_summary)
                        summary_text = " ".join(str(sentence) for sentence in summary)
                        st.markdown(summary_text if summary_text else "Not enough content to summarize effectively.")
                    else:
                        st.info("Not enough content to form a meaningful summary.")
                except Exception as e:
                    st.error(f"Error in Text Summarization: {e}. Ensure text is long enough for meaningful summary.")

        # --- Optional: Word Frequency Chart ---
        with st.expander("📊 Word Frequency Plot", expanded=False):
            if not tokens:
                st.warning("No words found after preprocessing for Word Frequency Plot.")
            else:
                try:
                    # Filter out non-alphabetic tokens and apply stopword removal again if not already
                    # (though it should be handled by `tokens` already)
                    filtered_words_for_freq = [w for w in tokens if w.isalpha()]
                    
                    freq_dist = nltk.FreqDist(filtered_words_for_freq)
                    common_words = freq_dist.most_common(15) # Show top 15 words

                    if common_words:
                        words, counts = zip(*common_words)
                        fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size
                        ax.bar(words, counts, color='teal')
                        ax.set_title("Top Most Frequent Words", fontsize=16)
                        ax.set_xlabel("Words", fontsize=12)
                        ax.set_ylabel("Frequency", fontsize=12)
                        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
                        plt.tight_layout() # Adjust layout to prevent labels overlapping
                        st.pyplot(fig)
                    else:
                        st.info("No frequent words found to plot.")
                except Exception as e:
                    st.error(f"Error generating Word Frequency Plot: {e}")