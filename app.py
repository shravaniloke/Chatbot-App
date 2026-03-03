from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ dataset
data = pd.read_csv("conv.csv")   # must contain columns: question, answer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    chat = ""

    if request.method == "POST":
        user_question = request.form["qts"].strip().lower()

        # Create text corpus
        texts = [user_question] + data["question"].str.lower().tolist()

        # Vectorization
        cv = CountVectorizer()
        vector = cv.fit_transform(texts)

        # Cosine Similarity
        similarity = cosine_similarity(vector)[0][1:]

        # Add similarity score
        data["score"] = similarity * 100

        # Sort by highest similarity
        result = data.sort_values(by="score", ascending=False)

        if result.iloc[0]["score"] < 10:
            msg = "🤖 KcBot: Sorry, I couldn't understand your query. Please contact Kamal Classes support."
        else:
            answer = result.iloc[0]["answer"]
            msg = f"🤖 KcBot: {answer}"

        chat = f"You: {user_question}\n{msg}"

        return render_template("home.html", chat=chat)

    return render_template("home.html")
    

if __name__ == "__main__":
    app.run(debug=True)