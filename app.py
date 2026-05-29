import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Title
st.title("📄 TextSense AI ")

st.write(
    "AI-Powered News & Text Classification System")
# Text Input
user_text = st.text_area(
    "Enter Text",
    height=180,
    placeholder="Paste a news article here or type your own text..."
)

# Prediction
if st.button("Predict"):

    if user_text.strip():

        text_vector = vectorizer.transform([user_text])

        prediction = model.predict(text_vector)

        category = prediction[0]

        category_emojis = {
            "business": "💼",
            "entertainment": "🎭",
            "politics": "🏛️",
            "sport": "⚽",
            "tech": "💻"
        }

        emoji = category_emojis.get(category, "📄")

        st.success(f"🎯 Predicted Category: {emoji} {category.title()}")

    else:
        st.warning("⚠️ Please enter some text.")

# Suggestions Section
st.subheader("💡 Quick Test Suggestions")
st.caption("Copy any example below and paste it into the textbox to test the classifier.")

st.info("""
1. The company reported a strong increase in quarterly profits after expanding into new markets.

2. Manchester United secured a dramatic victory after scoring in the final minutes of the match.

3. Researchers unveiled a new artificial intelligence system capable of understanding complex language.

4. The actor received critical acclaim for their performance in the latest blockbuster film.
""")

with st.expander("📖 Read More Suggestions"):

    st.markdown("""
**5.** Parliament members debated the proposed economic reforms during a lengthy session.

**6.** A major software company launched a new cloud computing platform for businesses.

**7.** The national team advanced to the tournament finals after a dominant performance.

**8.** The government announced a new policy aimed at reducing unemployment across the country.

**9.** The central bank raised interest rates in response to growing inflation concerns.

**10.** Scientists developed a new machine learning model capable of detecting diseases earlier.

**11.** The championship final attracted thousands of fans from across the country.

**12.** The movie broke box office records during its opening weekend.

**13.** Lawmakers discussed changes to taxation and public spending in parliament.

**14.** A startup introduced an innovative smartphone powered by advanced AI features.
""")