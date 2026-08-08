from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context, redirect, url_for, flash
import google.generativeai as genai
import os
from datetime import datetime
import json, random
import time
import re
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3
from contextlib import closing
import base64
from io import BytesIO
from PIL import Image
from googletrans import Translator
translator = Translator()
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask import redirect, url_for, session, flash
from flask import Flask, render_template, request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io

# ✅ Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Always resolve project files relative to this app.py file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Emotion descriptions
emotion_descriptions = {
    "Happy": "😊 You look happy. This suggests a positive, upbeat mood.",
    "Sad": "😢 You look sad. This may reflect feelings of disappointment, stress, or emotional pain.",
    "Angry": "😠 You look angry. You might be feeling frustration, injustice, or irritation.",
    "Surprise": "😲 You look surprised. Something unexpected may have just happened.",
    "Fear": "😨 You appear fearful. Possibly reacting to something uncertain or threatening.",
    "Disgust": "🤢 You show signs of disgust. Maybe something was upsetting or offensive.",
    "Calm": "😐 Your expression appears calm. No strong emotional cues were detected.",
    "Unknown": "🤔 No clear emotion detected from your face.",
    "Love": "❤️ You look in love. Your expression shows warmth, affection, and happiness."
}

last_detected_emotion = "Unknown"

# ✅ Load emotion detection model
emotion_model_path = os.path.join(BASE_DIR, 'models', 'fer2013_mini_XCEPTION.hdf5')
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ✅ Detect emotion from text
def detect_emotion(text):
    text_lower = text.lower()
    if "love" in text_lower or "lovely" in text_lower or "romantic" in text_lower:
        return "love"
    elif "happy" in text_lower:
        return "happy"
    elif "sad" in text_lower:
        return "sad"
    elif "angry" in text_lower:
        return "angry"
    else:
        return "calm"

def save_emotion_to_db(emotion):
    """Save detected emotion into the SQLite database"""
    conn = sqlite3.connect(os.path.join(BASE_DIR, "emotionsense.db"))
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emotion_history (emotion) VALUES (?)", (emotion,))
    conn.commit()
    conn.close()

def get_emotion_history(limit=50):
    """Fetch latest emotion history (default: last 50 records)"""
    conn = sqlite3.connect(os.path.join(BASE_DIR, "emotionsense.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT emotion, timestamp FROM emotion_history ORDER BY id DESC LIMIT ?", (limit,))
    data = cursor.fetchall()
    conn.close()
    return data

def translate_text(text, lang):
    if lang == 'en':  # Skip translation if English is selected
        return text
    try:
        return translator.translate(text, dest=lang).text
    except:
        return text  # fallback in case of error
    
analysis_results = []

def generate_pdf(results):
    buffer = io.BytesIO()  # In-memory PDF
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Text Analysis Result Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # Introduction
    intro_text = """
    The Text Sentiment Analysis module processes user-inputted text using NLP techniques. 
    The goal is to classify the underlying emotion expressed in the text into categories 
    such as Happy, Sad, Angry, or Neutral.
    """
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Table Data
    table_data = [["Input Text", "Predicted Emotion", "Confidence"]]
    for r in results:
        table_data.append([r["text"], r["emotion"], r["confidence"]])

    # Create Table
    table = Table(table_data, colWidths=[200, 120, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Observations
    obs_text = """
    <b>Observations:</b><br/>
    - Positive statements were strongly identified as Happy.<br/>
    - Negative, self-reflective texts were detected as Sad.<br/>
    - Aggressive or frustrated language was classified as Angry.<br/>
    - Informative or balanced sentences were categorized as Neutral.<br/>
    """
    elements.append(Paragraph(obs_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Conclusion
    conclusion_text = """
    <b>Conclusion:</b><br/>
    The Text Analysis module effectively distinguishes between different emotional tones in text.
    """
    elements.append(Paragraph(conclusion_text, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer
# ✅ Decorator to require login before accessing certain routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("⚠️ You need to log in first!", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ✅ Configure Google API (EmotionSense AI)
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# ✅ Sentiment analysis function
def analyze_with_emotionsense(text, analysis_type="basic"):
    """Analyze sentiment using EmotionSense AI"""
    try:
        if analysis_type == "basic":
            prompt = f"""
            You are **EmotionSense AI**, an emotional intelligence assistant.

            Analyze the sentiment of the following text and provide:
            1. Overall sentiment (Positive, Negative, or Neutral)
            2. Confidence score (0-100%)
            3. Brief explanation (2-3 sentences)

            Text: "{text}"

            Respond in this JSON format only:
            {{
                "sentiment": "Positive/Negative/Neutral",
                "confidence": 85,
                "explanation": "Your explanation here"
            }}
            """
        elif analysis_type == "detailed":
            prompt = f"""
            You are **EmotionSense AI**, an emotional intelligence assistant.

            Perform detailed emotional analysis of this text:
            1. Primary sentiment (Positive, Negative, Neutral)
            2. Emotional breakdown (joy, anger, sadness, fear, surprise, disgust) with scores 0-100
            3. Key phrases that influenced the sentiment
            4. Detailed explanation

            Text: "{text}"

            Respond in JSON format:
            {{
                "sentiment": "Primary sentiment",
                "confidence": 85,
                "emotions": {{
                    "joy": 70,
                    "anger": 10,
                    "sadness": 5,
                    "fear": 0,
                    "surprise": 15,
                    "disgust": 0
                }},
                "key_phrases": ["phrase1", "phrase2"],
                "explanation": "Detailed explanation"
            }}
            """
        elif analysis_type == "comparative":
            prompt = f"""
            You are **EmotionSense AI**, an emotional intelligence assistant.

            Perform comparative sentiment analysis using multiple approaches:
            1. Lexicon-based analysis
            2. Context-aware analysis
            3. Combined weighted result

            Text: "{text}"

            Respond in JSON format:
            {{
                "lexicon_based": {{
                    "sentiment": "Positive/Negative/Neutral",
                    "score": 0.75
                }},
                "context_aware": {{
                    "sentiment": "Positive/Negative/Neutral",
                    "score": 0.82
                }},
                "combined": {{
                    "sentiment": "Final sentiment",
                    "confidence": 85,
                    "explanation": "Why this is the final result"
                }}
            }}
            """
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # ✅ Extract JSON safely
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return {
                "sentiment": "Neutral",
                "confidence": 50,
                "explanation": "Unable to parse EmotionSense AI response properly"
            }
    except Exception as e:
        print(f"Error with EmotionSense AI: {str(e)}")
        return {
            "sentiment": "Error",
            "confidence": 0,
            "explanation": f"EmotionSense AI Error: {str(e)}"
        }
    
def init_db():
    with closing(sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS text_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    analysis_type TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS challenge_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT NOT NULL,
                    challenge TEXT NOT NULL,
                    response_text TEXT,
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emotion_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

def save_emotion(emotion, source):
    """Save detected emotion to SQLite database."""
    with closing(sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))) as conn:
        with conn:
            conn.execute(
                "INSERT INTO emotions (emotion, source) VALUES (?, ?)",
                (emotion, source)
            )


# ✅ ROUTES
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/api/docs')
@login_required
def api_docs():
    return render_template('api_docs.html')

@app.route('/analyze', methods=['GET'])
@login_required
def analyze_page():
    return render_template('analyze.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    text = request.form['text']

    # ✅ Detect language & translate to English if needed
    from googletrans import Translator
    translator = Translator()

    translated_text = text
    detected_lang = "en"

    try:
        detected = translator.detect(text)
        detected_lang = detected.lang
        if detected_lang != 'en':
            translated_text = translator.translate(text, src=detected_lang, dest='en').text
    except Exception as e:
        print("Translation error:", e)

    # ✅ Normalize detected emotion
    detected_emotion = detect_emotion(translated_text).lower()

    # ✅ Load suggestions
    suggestions_path = os.path.join(BASE_DIR, 'static', 'data', 'suggestions.json')
    with open(suggestions_path, 'r', encoding='utf-8') as f:
        suggestions = json.load(f)

    # ✅ Safe lookup
    emotion_data = suggestions.get(detected_emotion, suggestions.get("calm", {}))

    # ✅ Handle session counters
    if 'emotion_counters' not in session:
        session['emotion_counters'] = {}
    if detected_emotion not in session['emotion_counters']:
        session['emotion_counters'][detected_emotion] = 0

    index = session['emotion_counters'][detected_emotion]

    # ✅ Get suggestion data safely
    quote = emotion_data.get('quotes', ["No quotes available"])[index % len(emotion_data.get('quotes', ["No quotes available"]))]
    song = emotion_data.get('songs', ["No songs available"])[index % len(emotion_data.get('songs', ["No songs available"]))]
    video = emotion_data.get('videos', ["No videos available"])[index % len(emotion_data.get('videos', ["No videos available"]))]
    book = emotion_data.get('books', ["No book recommendation available"])[index % len(emotion_data.get('books', ["No book recommendation available"]))]

    # ✅ Update counter
    session['emotion_counters'][detected_emotion] += 1
    session.modified = True 

    # ✅ Save to DB
    import sqlite3
    from datetime import datetime

    conn = sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO text_analysis (text, sentiment, confidence, timestamp, analysis_type)
        VALUES (?, ?, ?, ?, ?)
    """, (
        text,  
        detected_emotion,
        100,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "basic"
    ))

    conn.commit()
    conn.close()

    # ✅ Send translated text to template too
    return render_template(
        'result.html',
        text=text,
        translated_text=translated_text if detected_lang != "en" else None,
        emotion=detected_emotion,
        quote=quote,
        song=song,
        video=video,
        book=book
    )
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    text = request.form["text"]

    # Example: Your text analysis logic
    emotion = "Happy"       # Replace with model output
    confidence = 92.4       # Replace with model output

    # Store in memory for PDF
    analysis_results.clear()  # Keep only current session result
    analysis_results.append({"text": text, "emotion": emotion, "confidence": f"{confidence}%"})

    return render_template("result.html", text=text, emotion=emotion, confidence=confidence)


@app.route('/history/text')
@login_required
def text_history():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, sentiment, confidence, timestamp, analysis_type FROM text_analysis ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    text_history = [{
        "id": row[0],
        "text": row[1],
        "sentiment": row[2],
        "confidence": row[3],
        "timestamp": row[4],
        "analysis_type": row[5]
    } for row in rows]

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('text_history.html', history=text_history)
    return redirect('/history')


@app.route('/history/face')
@login_required
def face_history():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT id, emotion, source, timestamp FROM emotions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    face_history = [{
        "id": row[0],
        "emotion": row[1],
        "source": row[2],
        "timestamp": row[3]
    } for row in rows]

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('face_history.html', history=face_history)
    return redirect('/history')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/models')
def models():
    return render_template('models.html')

# ✅ API endpoint for programmatic access
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400

    text = data['text'].strip()
    analysis_type = data.get('analysis_type', 'basic')
    emotion = data.get('emotion', '').strip()

    if emotion:
        text = f"(The user's facial expression indicates they are {emotion.lower()}.) " + text

    emotion_description = emotion_descriptions.get(emotion, emotion_descriptions["Unknown"])

    result = analyze_with_emotionsense(text, analysis_type)

    return jsonify({
        'success': True,
        'result': result,
        'analysis_type': analysis_type,
        'timestamp': datetime.now().isoformat(),
        'emotion': emotion,
        'emotion_description': emotion_description
    })

# ✅ Real-time emotion detection page
@app.route('/realtime-emotion')
@login_required
def realtime_emotion():
    return render_template('realtime_emotion.html')

# ✅ Webcam video feed
@app.route('/video_feed')
@login_required
def video_feed():
    def gen_frames():
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Try multiple camera indexes
        cap = None
        for idx in range(3):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"[INFO] Using camera index {idx}")
                break
        if not cap or not cap.isOpened():
            print("[ERROR] No camera found!")
            return

        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (64, 64))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)

                prediction = emotion_classifier.predict(face)
                emotion_label = emotion_labels[np.argmax(prediction)]

                # Save detected emotion
                save_emotion(emotion_label, "face")
                global last_detected_emotion
                last_detected_emotion = emotion_label

                # Draw rectangle + text (text only, no emojis)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    emotion_label,  # just the label
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/emotion_stream')
@login_required
def emotion_stream():
    def event_stream():
        global last_detected_emotion
        last_saved_emotion = None  # ✅ Track the last saved emotion
        while True:
            if last_detected_emotion and last_detected_emotion != last_saved_emotion:
                save_emotion_to_db(last_detected_emotion)
                last_saved_emotion = last_detected_emotion
            yield f"data: {last_detected_emotion}\n\n"
            time.sleep(1)
    return Response(stream_with_context(event_stream()), content_type='text/event-stream')

@app.route('/next_suggestion', methods=['POST'])
@login_required
def next_suggestion():
    data = request.get_json()
    detected_emotion = data.get('emotion', 'calm')

    suggestions_path = os.path.join(BASE_DIR, 'static', 'data', 'suggestions.json')
    with open(suggestions_path, 'r', encoding='utf-8') as f:
        suggestions = json.load(f)

    emotion_data = suggestions.get(detected_emotion, suggestions['calm'])

    # Initialize session counter
    if 'emotion_counters' not in session:
        session['emotion_counters'] = {}
    if detected_emotion not in session['emotion_counters']:
        session['emotion_counters'][detected_emotion] = 0

    index = session['emotion_counters'][detected_emotion]

    # Next suggestions
    quote = emotion_data['quotes'][index % len(emotion_data['quotes'])]
    song = emotion_data['songs'][index % len(emotion_data['songs'])]
    video = emotion_data['videos'][index % len(emotion_data['videos'])]
    
    # ✅ Get 1 book at a time
    books = emotion_data.get('books', [])
    book = books[index % len(books)] if books else None

    session['emotion_counters'][detected_emotion] += 1
    session.modified = True

    return jsonify({
        'quote': quote,
        'song': song,
        'video': video,
        'book': book  # send only 1 book
    })

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard to show emotion statistics."""
    with closing(sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Count how many times each emotion appears
        cursor.execute("SELECT emotion, COUNT(*) as count FROM emotions GROUP BY emotion")
        emotion_counts = cursor.fetchall()

        # Get timeline (for line chart)
        cursor.execute("""
            SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour, emotion, COUNT(*) as count
            FROM emotions
            GROUP BY hour, emotion
            ORDER BY hour
        """)
        timeline_data = cursor.fetchall()

    # ✅ Add emoji + description for each emotion (used in card display)
    emotion_cards = {
        'Angry': {'emoji': '😠', 'desc': 'Anger detected — try some calm!'},
        'Disgust': {'emoji': '🤢', 'desc': 'Feeling disgusted — take a break!'},
        'Fear': {'emoji': '😨', 'desc': 'Fear sensed — stay strong!'},
        'Happy': {'emoji': '😄', 'desc': 'Joyful vibes all around!'},
        'Neutral': {'emoji': '😐', 'desc': 'Balanced and calm mood.'},
        'Sad': {'emoji': '😢', 'desc': 'Sadness detected — sending hugs!'},
        'Surprise': {'emoji': '😲', 'desc': 'Something surprising happened!'},
        'Love': {'emoji': '😍', 'desc': 'Love in the air — beautiful!'}
    }

    return render_template("dashboard.html",
                           emotion_counts=emotion_counts,
                           timeline_data=timeline_data,
                           emotion_cards=emotion_cards)

# ✅ NEW ➡️ Handle "Mood Selfie" snapshot
@app.route('/analyze_face', methods=['POST'])
@login_required
def analyze_face():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # ✅ Remove the base64 header
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image_bytes = base64.b64decode(image_data)

        # ✅ Open the image
        image = Image.open(BytesIO(image_bytes)).convert('L')  # grayscale
        image = image.resize((64, 64))  # match model input

        img_array = np.array(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        # ✅ Predict emotion using your loaded model
        prediction = emotion_classifier.predict(img_array)
        emotion_label = emotion_labels[np.argmax(prediction)]

        # ✅ Save detected emotion to DB
        save_emotion(emotion_label, "selfie")

        return jsonify({"face_emotion": emotion_label})

    except Exception as e:
        print("❌ Error in /analyze_face:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/get-face-suggestions", methods=["POST"])
@login_required
def get_face_suggestions():
    # ✅ Get emotion from request
    data = request.get_json()
    emotion = data.get("emotion", "calm").lower()

    # ✅ Build absolute path to static/data/face_suggestions.json
    json_path = os.path.join(app.root_path, "static", "data", "face_suggestions.json")

    # ✅ Read JSON file safely
    with open(json_path, "r", encoding="utf-8") as f:
        suggestions = json.load(f)

    # ✅ Pick random tip, gif, and challenge based on emotion
    if emotion in suggestions:
        tip = random.choice(suggestions[emotion]["tips"])
        gif = random.choice(suggestions[emotion]["gifs"])
        challenge = random.choice(suggestions[emotion]["challenges"])
    else:
        tip = "Take a deep breath 🌿"
        gif = "https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif"
        challenge = "Write one thing you’re grateful for ✨"

    return jsonify({
        "tip": tip,
        "gif": gif,
        "challenge": challenge
    })
    
@app.route("/emotion-history")
@login_required
def emotion_history():
    history = get_emotion_history(50)  # last 50 entries
    return {"history": [{"emotion": e, "timestamp": t} for e, t in history]}

@app.route('/set_language/<lang>')
def set_language(lang):
    session['lang'] = lang
    return redirect(request.referrer or url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, hashed_pw))
            conn.commit()
            flash("✅ Registration successful! Please login.", "success")
        except sqlite3.IntegrityError:
            flash("❌ Username or Email already exists!", "danger")
            conn.close()
            return redirect(url_for('register'))

        conn.close()
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash("✅ Welcome back, {}!".format(user[1]), "success")

            # 🔄 CHANGE HERE: send user to home instead of dashboard
            return redirect(url_for('home'))  

        else:
            flash("❌ Invalid email or password!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash("✅ You have logged out.", "info")
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "emotion_data.db"))
    cursor = conn.cursor()

    # Face history
    cursor.execute("SELECT id, emotion, source, timestamp FROM emotions ORDER BY id DESC")
    face_rows = cursor.fetchall()
    face_history = [{
        "id": row[0],
        "emotion": row[1],
        "source": row[2],
        "timestamp": row[3]
    } for row in face_rows]

    # Text history
    cursor.execute("SELECT id, text, sentiment, confidence, timestamp, analysis_type FROM text_analysis ORDER BY id DESC")
    text_rows = cursor.fetchall()
    text_history = [{
        "id": row[0],
        "text": row[1],
        "sentiment": row[2],
        "confidence": row[3],
        "timestamp": row[4],
        "analysis_type": row[5]
    } for row in text_rows]

    conn.close()

    return render_template("history.html", face_history=face_history, text_history=text_history)

# ✅ Run Flask app
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
