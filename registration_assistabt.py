import re
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------
# NLTK SETUP
# -------------------------------------------------

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# -------------------------------------------------
# FLASK APP
# -------------------------------------------------

app = Flask(__name__)


# -------------------------------------------------
# AVAILABLE COURSES
# -------------------------------------------------

courses = [
    "Python Programming",
    "Data Science",
    "Web Development"
]


# -------------------------------------------------
# REGISTRATION ASSISTANT
# -------------------------------------------------

class RegistrationAssistant:

    def __init__(self):

        # Student information
        self.user_data = {}

        # NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Chatbot intents
        self.intents = self.define_intents()


    # -------------------------------------------------
    # INTENTS
    # -------------------------------------------------

    def define_intents(self):

        return {

            "greeting": {
                "patterns": [
                    "hi",
                    "hello",
                    "hey",
                    "good morning",
                    "good afternoon",
                    "good evening"
                ],
                "responses": [
                    "Hello! Welcome to the AI Registration Assistant. How can I help you?"
                ]
            },

            "register": {
                "patterns": [
                    "register",
                    "apply",
                    "sign up",
                    "join",
                    "registration",
                    "register me"
                ],
                "responses": [
                    "Great! I can help you with registration. Please provide your details."
                ]
            },

            "eligibility": {
                "patterns": [
                    "eligibility",
                    "eligible",
                    "qualify",
                    "qualification",
                    "am i eligible",
                    "eligibility criteria"
                ],
                "responses": [
                    "To be eligible, you must be at least 18 years old and have at least 50% marks."
                ]
            },

            "course": {
                "patterns": [
                    "course",
                    "courses",
                    "program",
                    "programs",
                    "available",
                    "available courses"
                ],
                "responses": [
                    "Available courses are Python Programming, Data Science and Web Development."
                ]
            },

            "guidance": {
                "patterns": [
                    "how do i register",
                    "registration process",
                    "registration steps",
                    "how to register",
                    "guide",
                    "steps"
                ],
                "responses": [
                    "First check your eligibility, select a course, prepare your documents, complete the registration form and submit it."
                ]
            },

            "documents": {
                "patterns": [
                    "document",
                    "documents",
                    "certificate",
                    "identity proof",
                    "proof",
                    "required documents"
                ],
                "responses": [
                    "You may need academic certificates, identity proof and other required documents."
                ]
            },

            "status": {
                "patterns": [
                    "status",
                    "registration status",
                    "application status",
                    "check status"
                ],
                "responses": [
                    "You can check your registration status after submitting your registration."
                ]
            },

            "help": {
                "patterns": [
                    "help",
                    "support",
                    "assist",
                    "guide"
                ],
                "responses": [
                    "I can help you with registration, eligibility, courses, documents and registration status."
                ]
            },

            "thanks": {
                "patterns": [
                    "thank",
                    "thanks",
                    "thank you",
                    "appreciate"
                ],
                "responses": [
                    "You're welcome! Is there anything else I can help you with?"
                ]
            },

            "goodbye": {
                "patterns": [
                    "bye",
                    "goodbye",
                    "exit",
                    "quit"
                ],
                "responses": [
                    "Thank you for using the AI Registration Assistant. Goodbye!"
                ]
            }
        }


    # -------------------------------------------------
    # TEXT PREPROCESSING
    # -------------------------------------------------

    def preprocess_text(self, text):

        # Convert to lowercase
        text = text.lower()

        # Remove unwanted characters
        text = re.sub(r"[^a-zA-Z0-9\s@.]", "", text)

        # Tokenization
        tokens = nltk.word_tokenize(text)

        # Remove stopwords and perform lemmatization
        processed_tokens = []

        for token in tokens:

            if token not in self.stop_words:

                lemma = self.lemmatizer.lemmatize(token)

                processed_tokens.append(lemma)

        return processed_tokens


    # -------------------------------------------------
    # INTENT CLASSIFICATION
    # -------------------------------------------------

    def classify_intent(self, text):

        text_lower = text.lower()

        tokens = self.preprocess_text(text)

        # Check exact multi-word patterns first
        for intent, data in self.intents.items():

            for pattern in data["patterns"]:

                if " " in pattern:

                    if pattern.lower() in text_lower:

                        return intent


        # Check individual words
        for intent, data in self.intents.items():

            for pattern in data["patterns"]:

                pattern_words = pattern.lower().split()

                for word in pattern_words:

                    if word in tokens:

                        return intent


        return "unknown"


    # -------------------------------------------------
    # ENTITY EXTRACTION
    # -------------------------------------------------

    def extract_entities(self, text):

        entities = {}


        # NAME
        name_match = re.search(
            r"\b(?:my name is|i am|i'm|name is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
            text,
            re.IGNORECASE
        )

        if name_match:

            entities["name"] = name_match.group(1).strip()


        # EMAIL
        email_match = re.search(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text
        )

        if email_match:

            entities["email"] = email_match.group()


        # COURSE
        text_lower = text.lower()

        for course in courses:

            if course.lower() in text_lower:

                entities["course"] = course


        return entities


    # -------------------------------------------------
    # RESPONSE GENERATION
    # -------------------------------------------------

    def get_response(self, user_input):

        # Extract entities
        entities = self.extract_entities(user_input)


        # Save student information
        for key, value in entities.items():

            self.user_data[key] = value


        # Classify intent
        intent = self.classify_intent(user_input)


        # Unknown intent
        if intent == "unknown":

            return (
                "I'm not sure I understood. "
                "You can ask me about registration, eligibility, "
                "courses, documents or registration status."
            )


        # Get normal response
        response = self.intents[intent]["responses"][0]


        # Name entity
        if "name" in entities:

            response = (
                "Nice to meet you, "
                + entities["name"]
                + "! What would you like to know about registration?"
            )


        # Email entity
        elif "email" in entities:

            response = (
                "Thank you! Your email "
                + entities["email"]
                + " has been recorded."
            )


        # Course entity
        elif "course" in entities:

            response = (
                "You selected "
                + entities["course"]
                + ". I can help you with its eligibility and registration."
            )


        return response


# -------------------------------------------------
# CREATE ASSISTANT
# -------------------------------------------------

assistant = RegistrationAssistant()


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -------------------------------------------------
# CHAT API
# -------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "response": "Please enter a message."
            })


        user_message = data.get("message", "").strip()


        if not user_message:

            return jsonify({
                "response": "Please enter a message."
            })


        response = assistant.get_response(user_message)


        return jsonify({
            "response": response
        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "response": "Sorry, something went wrong while processing your message."
        }), 500


# -------------------------------------------------
# RUN SERVER
# -------------------------------------------------

if __name__ == "__main__":

    print("------------------------------------------")
    print("AI REGISTRATION ASSISTANT")
    print("------------------------------------------")
    print("Server running at:")
    print("http://127.0.0.1:8001")
    print("------------------------------------------")

    app.run(
        host="127.0.0.1",
        port=8001,
        debug=False
    )
