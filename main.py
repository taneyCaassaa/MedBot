from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access
app.config['SECRET_KEY'] = os.urandom(24)  # Required for sessions
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)  # Enable session management

# Load API key
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Validation helpers
OFFENSIVE = {"fuck", "shit", "bitch", "pussy", "dick", "ass", "chutiya"}

def is_offensive(text):
    if not isinstance(text, str):
        return False
    return any(word in text.lower() for word in OFFENSIVE)

def validate_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        return False, f"{field_name} must be a non-empty string"
    if is_offensive(value):
        return False, f"{field_name} contains offensive words"
    return True, None

def validate_number(value, field_name, min_val=None, max_val=None):
    try:
        num = int(value)
        if min_val is not None and num < min_val:
            return False, f"{field_name} must be at least {min_val}"
        if max_val is not None and num > max_val:
            return False, f"{field_name} must be at most {max_val}"
        return True, None
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid number"

def validate_choice(value, field_name, options):
    if not isinstance(value, str):
        return False, f"{field_name} must be a string"
    if is_offensive(value):
        return False, f"{field_name} contains offensive words"
    if value.lower() not in [opt.lower() for opt in options]:
        return False, f"{field_name} must be one of: {', '.join(options)}"
    return True, None

# Question sequence and validation rules
QUESTIONS = [
    {"key": "name", "prompt": "What is your name?", "validator": validate_string, "kwargs": {}},
    {"key": "age", "prompt": "How old are you?", "validator": validate_number, "kwargs": {"min_val": 0, "max_val": 120}},
    {"key": "sex", "prompt": "What is your biological sex?", "validator": validate_choice, "kwargs": {"options": ["Male", "Female", "Other"]}},
    {"key": "height_cm", "prompt": "What is your height in cm?", "validator": validate_number, "kwargs": {"min_val": 50, "max_val": 300}},
    {"key": "weight_kg", "prompt": "What is your weight in kg?", "validator": validate_number, "kwargs": {"min_val": 2, "max_val": 500}},
    {"key": "has_history", "prompt": "Do you have any diagnosed medical conditions?", "validator": validate_choice, "kwargs": {"options": ["Yes", "No"]}},
    {"key": "has_meds", "prompt": "Are you currently taking any medications?", "validator": validate_choice, "kwargs": {"options": ["Yes", "No"]}},
    {"key": "has_allergy", "prompt": "Do you have any allergies?", "validator": validate_choice, "kwargs": {"options": ["Yes", "No"]}},
    {"key": "fam_hist", "prompt": "Do any medical conditions run in your family?", "validator": validate_choice, "kwargs": {"options": ["Yes", "No"]}},
    {"key": "lifestyle", "prompt": "Do you smoke, drink alcohol, and how active are you physically?", "validator": validate_string, "kwargs": {}},
    {"key": "diet", "prompt": "How would you describe your diet?", "validator": validate_string, "kwargs": {}},
    {"key": "stress_sleep", "prompt": "How would you rate your stress level?", "validator": validate_string, "kwargs": {}},
    {"key": "sleep", "prompt": "On average, how many hours of sleep do you get per night?", "validator": validate_number, "kwargs": {"min_val": 0, "max_val": 10}},
    {"key": "main_symptom", "prompt": "What’s the main symptom that’s bothering you?", "validator": validate_string, "kwargs": {}},
    {"key": "duration", "prompt": "How long have you been experiencing this symptom?", "validator": validate_string, "kwargs": {}},
    {"key": "severity", "prompt": "On a scale from 1 to 10, how severe is it?", "validator": validate_number, "kwargs": {"min_val": 1, "max_val": 10}},
    {"key": "pattern", "prompt": "How would you describe the pattern?", "validator": validate_choice, "kwargs": {"options": ["Constant", "Intermittent", "Worsening", "Other"]}},
    {"key": "triggers", "prompt": "Does anything trigger or worsen it?", "validator": validate_string, "kwargs": {}},
    {"key": "relief", "prompt": "Have you tried anything that helps relieve it?", "validator": validate_string, "kwargs": {}},
    {"key": "associated", "prompt": "Are you experiencing any other symptoms?", "validator": validate_string, "kwargs": {}}
]

# Conditional questions based on decision tree
CONDITIONAL_QUESTIONS = {
    "has_history": {"value": "Yes", "question": {"key": "history", "prompt": "Please list your diagnosed conditions", "validator": validate_string, "kwargs": {}}},
    "has_meds": {"value": "Yes", "question": {"key": "medications", "prompt": "Please list your current medications", "validator": validate_string, "kwargs": {}}},
    "has_allergy": {"value": "Yes", "question": {"key": "allergies", "prompt": "Please list your allergies", "validator": validate_string, "kwargs": {}}},
    "fam_hist": {"value": "Yes", "question": {"key": "family_history", "prompt": "Which conditions run in your family?", "validator": validate_string, "kwargs": {}}},
    "pattern": {"value": "Other", "question": {"key": "pattern_description", "prompt": "Please describe the pattern", "validator": validate_string, "kwargs": {}}}
}

# Generate structured report
def generate_report(data):
    template = PromptTemplate(
        input_variables=["info"],
        template="""
You are a helpful medical assistant. Based only on the patient information,
generate a structured health assessment.

Rules:
- Mention only real, known medical conditions (no placeholders).
- Suggest differential diagnoses based on age, sex, history, lifestyle, and symptoms.
- If info is insufficient, say so.
- Never give a definitive diagnosis. Use "possible", "may indicate", "consider".
- Keep tone professional but patient-friendly.
- End with a clear disclaimer: "This is not medical advice. Please consult a licensed healthcare professional."

Patient Info:
{info}

Format:
===============================
🧾 Medical Assessment Report
-------------------------------
**Patient Summary**
**Possible Conditions**
**Warning Signs**
**Self-care Recommendations**
**Prevention Measures**
**Recommended Specialists**
===============================
"""
    )

    prompt = template.format(info=data)
    report = llm.predict(prompt)
    return report

# API endpoints
@app.route('/api/start', methods=['POST'])
def start_session():
    session.clear()
    session['session_id'] = str(uuid.uuid4())
    session['answers'] = {}
    session['question_index'] = 0
    return jsonify({
        "message": "Hi, I'm your health assistant. Let's start.",
        "question": QUESTIONS[0]["prompt"],
        "key": QUESTIONS[0]["key"],
        "type": "string" if QUESTIONS[0]["validator"] == validate_string else "number" if QUESTIONS[0]["validator"] == validate_number else "choice",
        "options": QUESTIONS[0]["kwargs"].get("options", []),
        "min_val": QUESTIONS[0]["kwargs"].get("min_val"),
        "max_val": QUESTIONS[0]["kwargs"].get("max_val")
    })

@app.route('/api/answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    if not data or 'answer' not in data or 'key' not in data:
        return jsonify({"error": "Missing answer or key"}), 400

    answer = data['answer']
    key = data['key']
    if 'answers' not in session or 'question_index' not in session:
        return jsonify({"error": "Session not started. Please start a new session."}), 400

    # Find the current question
    current_index = session['question_index']
    if current_index >= len(QUESTIONS) and key not in [q['key'] for q in CONDITIONAL_QUESTIONS.values()]:
        return jsonify({"error": "No more questions"}), 400

    # Validate the answer
    question = None
    if current_index < len(QUESTIONS) and QUESTIONS[current_index]['key'] == key:
        question = QUESTIONS[current_index]
    else:
        for cond_key, cond_info in CONDITIONAL_QUESTIONS.items():
            if cond_info['question']['key'] == key:
                question = cond_info['question']
                break

    if not question:
        return jsonify({"error": "Invalid question key"}), 400

    valid, error = question['validator'](answer, key, **question['kwargs'])
    if not valid:
        return jsonify({"error": error}), 400

    # Store the answer
    session['answers'][key] = answer
    session.modified = True

    # Check for conditional questions
    next_question = None
    if key in CONDITIONAL_QUESTIONS and answer == CONDITIONAL_QUESTIONS[key]['value']:
        next_question = CONDITIONAL_QUESTIONS[key]['question']
    else:
        # Move to the next question
        session['question_index'] = current_index + 1
        session.modified = True
        if session['question_index'] < len(QUESTIONS):
            next_question = QUESTIONS[session['question_index']]

    # Set defaults for conditional fields if not applicable
    if key == "has_history" and answer == "No":
        session['answers']['history'] = "None"
    if key == "has_meds" and answer == "No":
        session['answers']['medications'] = "None"
    if key == "has_allergy" and answer == "No":
        session['answers']['allergies'] = "None"
    if key == "fam_hist" and answer == "No":
        session['answers']['family_history'] = "None"
    if key == "pattern" and answer == "Other":
        session['answers']['pattern'] = session['answers'].get('pattern_description', answer)
    elif key == "pattern_description":
        session['answers']['pattern'] = answer

    if next_question:
        return jsonify({
            "question": next_question['prompt'],
            "key": next_question['key'],
            "type": "string" if next_question['validator'] == validate_string else "number" if next_question['validator'] == validate_number else "choice",
            "options": next_question['kwargs'].get("options", []),
            "min_val": next_question['kwargs'].get("min_val"),
            "max_val": next_question['kwargs'].get("max_val")
        })
    else:
        # All questions answered, generate report
        report = generate_report(session['answers'])
        session.clear()  # Clear session after report
        return jsonify({"report": report})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)





# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv

# # Load API key
# load_dotenv(".env")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(
#     model="gpt-4o-mini", 
#     temperature=0,
#     api_key=OPENAI_API_KEY
# )

# # Validation helpers
# OFFENSIVE = {"fuck", "shit", "bitch", "pussy", "dick", "ass", "chutiya"}

# def is_offensive(text):
#     return any(word in text.lower() for word in OFFENSIVE)

# def ask_free_text(prompt):
#     while True:
#         ans = input(f"🤖 {prompt}\nYou: ")
#         if is_offensive(ans):
#             print("❌ Offensive words not allowed. Try again.")
#             continue
#         if ans.strip() == "":
#             print("❌ Please provide a valid answer.")
#             continue
#         return ans.strip()

# def ask_number(prompt, min_val=None, max_val=None):
#     while True:
#         ans = input(f"🤖 {prompt}\nYou: ")
#         if not ans.isdigit():
#             print("❌ Please enter a valid number.")
#             continue
#         num = int(ans)
#         if min_val is not None and num < min_val:
#             print(f"❌ Must be at least {min_val}")
#             continue
#         if max_val is not None and num > max_val:
#             print(f"❌ Must be at most {max_val}")
#             continue
#         return num

# def ask_choice(prompt, options):
#     options_str = ", ".join(options)
#     while True:
#         ans = input(f"🤖 {prompt} ({options_str})\nYou: ").lower()
#         if is_offensive(ans):
#             print("❌ Offensive words not allowed. Try again.")
#             continue
#         if ans in [opt.lower() for opt in options]:
#             return ans.capitalize()
#         print(f"❌ Please choose one of: {options_str}")

# # Collect patient data with decision-tree flow
# def collect_patient_data():
#     data = {}
#     print("🤖 Hi, I'm your health assistant. Let's start.")

#     data["name"] = ask_free_text("What is your name?")
#     data["age"] = ask_number("How old are you?", min_val=0, max_val=120)
#     data["sex"] = ask_choice("What is your biological sex?", ["Male", "Female", "Other"])
#     data["height_cm"] = ask_number("What is your height in cm?", min_val=50, max_val=300)
#     data["weight_kg"] = ask_number("What is your weight in kg?", min_val=2, max_val=500)

#     # Decision tree: history
#     has_history = ask_choice("Do you have any diagnosed medical conditions?", ["Yes", "No"])
#     if has_history == "Yes":
#         data["history"] = ask_free_text("Please list your diagnosed conditions")
#     else:
#         data["history"] = "None"

#     has_meds = ask_choice("Are you currently taking any medications?", ["Yes", "No"])
#     if has_meds == "Yes":
#         data["medications"] = ask_free_text("Please list your current medications")
#     else:
#         data["medications"] = "None"

#     has_allergy = ask_choice("Do you have any allergies?", ["Yes", "No"])
#     if has_allergy == "Yes":
#         data["allergies"] = ask_free_text("Please list your allergies")
#     else:
#         data["allergies"] = "None"

#     fam_hist = ask_choice("Do any medical conditions run in your family?", ["Yes", "No"])
#     if fam_hist == "Yes":
#         data["family_history"] = ask_free_text("Which conditions run in your family?")
#     else:
#         data["family_history"] = "None"

#     data["lifestyle"] = ask_free_text("Do you smoke, drink alcohol, and how active are you physically?")
#     data["diet"] = ask_free_text("How would you describe your diet?")
#     data["stress_sleep"] = ask_free_text("How would you rate your stress level?")
#     data["sleep"] = ask_number("On average, how many hours of sleep do you get per night?", min_val=0, max_val=10)
#     data["main_symptom"] = ask_free_text("What’s the main symptom that’s bothering you?")
#     data["duration"] = ask_free_text("How long have you been experiencing this symptom?")
#     data["severity"] = ask_number("On a scale from 1 to 10, how severe is it?", min_val=1, max_val=10)
#     data["pattern"] = ask_choice("How would you describe the pattern?", ["Constant", "Intermittent", "Worsening", "Other"])
#     if data["pattern"] == "Other":
#         data["pattern"] = ask_free_text("Please describe the pattern")

#     data["triggers"] = ask_free_text("Does anything trigger or worsen it?")
#     data["relief"] = ask_free_text("Have you tried anything that helps relieve it?")
#     data["associated"] = ask_free_text("Are you experiencing any other symptoms?")

#     return data

# # Generate structured report
# def generate_report(data):
#     template = PromptTemplate(
#         input_variables=["info"],
#         template="""
# You are a helpful medical assistant. Based only on the patient information,
# generate a structured health assessment.

# Rules:
# - Mention only real, known medical conditions (no placeholders).
# - Suggest differential diagnoses based on age, sex, history, lifestyle, and symptoms.
# - If info is insufficient, say so.
# - Never give a definitive diagnosis. Use "possible", "may indicate", "consider".
# - Keep tone professional but patient-friendly.
# - End with a clear disclaimer: "This is not medical advice. Please consult a licensed healthcare professional."

# Patient Info:
# {info}

# Format:
# ===============================
# 🧾 Medical Assessment Report
# -------------------------------
# **Patient Summary**
# **Possible Conditions**
# **Warning Signs**
# **Self-care Recommendations**
# **Prevention Measures**
# **Recommended Specialists**
# ===============================
# """
#     )

#     prompt = template.format(info=data)
#     report = llm.predict(prompt)
#     print("\n📋 Here is your assessment report:\n")
#     print(report)

# # Main
# if __name__ == "__main__":
#     patient_data = collect_patient_data()
#     generate_report(patient_data)
