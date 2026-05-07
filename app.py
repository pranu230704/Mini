from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.parse
import tldextract
import xgboost as xgb
import re

app = Flask(__name__)
CORS(app)

# === Load All Models (Phishing, SQL Injection, XSS) ===
try:
    # ---- Load Phishing Model (XGBoost) ----
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/phishing_model.json')

    # ---- Load SQL Injection Model (BiLSTM + Attention / CNN-GRU) ----
    sql_model = load_model('models/sql_injection_model.h5')

    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    print("✔ Phishing and SQL Injection models loaded successfully.")

    # ---- Load XSS Detection Model (Random Forest + TF-IDF) ----
    try:
        xss_model = joblib.load('models/xss_random_forest_model.pkl')

        with open('models/xss_tfidf_vectorizer.pkl', 'rb') as f:
            xss_vectorizer = pickle.load(f)
        
        print("✔ XSS models loaded successfully.")
    except Exception as xss_error:
        print(f"⚠ XSS models failed to load: {xss_error}")
        print("⚠ XSS detection will be unavailable.")
        xss_model = None
        xss_vectorizer = None

except Exception as e:
    print(f"[Critical Model Load Error] {e}")
    raise

# === Feature Extraction for Phishing Detection ===
def extract_url_features(url):
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        ext = tldextract.extract(url)

        features = [
            len(url), int(url.startswith('https://')), domain.count('.'),
            int('@' in url), int('//' in url[8:]), int('-' in domain),
            len(ext.subdomain.split('.')), int(any(part.isdigit() for part in domain.split('.'))),
            len(domain), int(domain.lower() != ext.domain.lower()),
            len(path.split('/')), int('%' in url), int('&' in url), int('=' in url),
            int('?' in url), int('#' in url), int('~' in url), int(',' in url), int('+' in url),
            int('_' in url), int(';' in url), int('$' in url), int('!' in url), int('*' in url),
            int('(' in url), int(')' in url), int('|' in url), int('^' in url), int('{' in url),
            int('}' in url)
        ]

        # Keywords
        keywords = ['login', 'signin', 'bank', 'account', 'verify', 'secure', 'update', 'webscr', 'password', 'confirm']
        features.extend([int(keyword in url.lower()) for keyword in keywords])

        # TLDs
        tlds = ['.com', '.net', '.org', '.info', '.biz', '.ru', '.uk', '.in']
        features.extend([int(url.endswith(tld)) for tld in tlds])

        # Remove these placeholders:
        # features.extend([0]*30)

        features.append(int(len(query) > 0))
        features.append(int('php' in path.lower()))
        features.append(int('asp' in path.lower()))
        features.append(int('js' in path.lower()))

        # Pad zeros only if feature length < expected by model
        while len(features) < xgb_model.n_features_in_:
            features.append(0)

        return np.array(features)
    except Exception as e:
        print(f"[Feature Extraction Error] {e}")
        return np.zeros(xgb_model.n_features_in_)

def detect_phishing_patterns(url):
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    ext = tldextract.extract(url)
    patterns = []

    if '@' in url:
        patterns.append("URL contains @ symbol")
    if '//' in url[8:]:
        patterns.append("Possible redirect using //")
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain.split(':')[0]):
        patterns.append("IP address used as domain")
    if len(url) > 120:
        patterns.append("Very long URL")
    if domain.count('.') >= 4:
        patterns.append("Too many subdomains")
    if '%' in url:
        patterns.append("Encoded characters in URL")

    phishing_words = ['login', 'signin', 'bank', 'account', 'verify', 'secure', 'update', 'webscr', 'password', 'confirm']
    has_sensitive_word = any(word in url.lower() for word in phishing_words)
    has_risky_context = parsed.scheme != 'https' or bool(query) or '-' in domain or len(ext.subdomain.split('.')) > 1

    if has_sensitive_word and has_risky_context:
        patterns.append("Sensitive keyword in risky URL context")

    return patterns

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/phishing', methods=['POST'])
def predict_phishing():
    data = request.get_json()
    url = data.get('url', '')

    if not url.startswith('http'):
        return jsonify({'error': 'Invalid URL'}), 400

    phishing_patterns = detect_phishing_patterns(url)

    features = extract_url_features(url)
    if len(features) != xgb_model.n_features_in_:
        return jsonify({'error': 'Feature mismatch'}), 500
        
    pred_probs = xgb_model.predict_proba([features])[0]

    phishing_prob = pred_probs[0]  # class 0 = phishing
    safe_prob = pred_probs[1]      # class 1 = safe

    threshold = 0.3  # set based on what works best empirically

    if phishing_prob > threshold:
        pred_class = 0  # phishing
        confidence = float(phishing_prob)
    else:
        pred_class = 1  # safe
        confidence = float(safe_prob)

    if phishing_patterns:
        pred_class = 0
        confidence = max(float(phishing_prob), 0.95)
    else:
        pred_class = 1
        confidence = max(float(safe_prob), 0.95)

    print("[DEBUG] URL:", url)
    print("[DEBUG] Feature Vector Length:", len(features))
    print("[DEBUG] Feature Vector:", features)
    print("[DEBUG] Prediction Probabilities:", pred_probs)
    print("[DEBUG] Phishing Patterns:", phishing_patterns)
    print("[DEBUG] Prediction Class:", pred_class)
    print("[DEBUG] Confidence Score:", confidence)
    print(f"[DEBUG] Final prediction: {pred_class} ({'Phishing' if pred_class == 0 else 'Safe'})")

    label_map = {0: "Phishing", 1: "Safe"}

    return jsonify({
        'prediction': pred_class,
        'confidence': confidence,
       # 'features': [f"feature_{i}: {val}" for i, val in enumerate(features)],
        })

# === Improved Rule-based Signature Detection ===
def is_malicious_pattern(query: str) -> list:
    q = query.lower()
    patterns = []

    if '1=1' in q and re.search(r"\bor\b|\band\b", q):
        patterns.append("Always true condition (1=1)")
    if re.search(r"--|#", q) and not re.search(r"like\s+'%--%'", q):
        patterns.append("SQL comment detected")
    if re.search(r"\bunion\b", q) and "select" in q:
        patterns.append("UNION keyword detected")
    if re.search(r";\s*(drop|delete|insert|update|select)", q):
        patterns.append("Stacked query or destructive command")
    if 'sleep(' in q or 'waitfor' in q:
        patterns.append("Time delay function")
    if 'exec(' in q or 'xp_' in q:
        patterns.append("Command execution")
    if re.search(r"'\s*(or|and)\s+'?\w+'?\s*=\s*'?\w+", q):
        patterns.append("Boolean SQL injection pattern")
    if re.search(r"\b(drop|truncate|alter)\s+table\b", q):
        patterns.append("Destructive SQL command")

    return patterns

def looks_like_plain_text(query: str) -> bool:
    sql_words = r"\b(select|insert|update|delete|drop|union|where|from|values|exec|sleep|waitfor|table)\b"
    sql_symbols = r"['\";=()#-]"
    return not re.search(sql_words, query.lower()) and not re.search(sql_symbols, query)

# === SQL Injection Detection Endpoint ===
@app.route('/predict/sql', methods=['POST'])
def predict_sql():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        cleaned = ' '.join(query.strip().lower().split())
        patterns = is_malicious_pattern(cleaned)

        # === Confidence Boost for Legitimate Patterns ===
        # Strong confidence boost for known-safe queries
        safe_patterns = [
            r"^select\s+\*\s+from\s+\w+\s+where\s+\w+\s*=\s*['\"].+['\"]$",
            r"^select\s+[\w\s,.*]+\s+from\s+\w+(\s+where\s+[\w\s.=<>!'\"%]+)?$",
            r"^insert\s+into\s+\w+\s*\(.+\)\s+values\s*\(.+\)$",
            r"^update\s+\w+\s+set\s+.+\s+where\s+.+$",
            r"^delete\s+from\s+\w+\s+where\s+.+$"
        ]

        if patterns:
            print(f"Cleaned Query: {cleaned}")
            print("[SQL Pattern Detected] Forcing malicious.")
            return jsonify({
                'prediction': 1,
                'confidence': 0.95,
                'patterns': patterns,
                'preprocessed': cleaned
            })

        if looks_like_plain_text(cleaned):
            print(f"Cleaned Query: {cleaned}")
            print("[Plain Text Detected] Forcing legitimate.")
            return jsonify({
                'prediction': 0,
                'confidence': 0.95,
                'patterns': patterns,
                'preprocessed': cleaned
            })

        for pattern in safe_patterns:
            if re.fullmatch(pattern, cleaned, flags=re.IGNORECASE):
                print("[Safe Pattern Detected] Forcing low confidence.")
                return jsonify({
                    'prediction': 0,
                    'confidence': 0.98,
                    'patterns': patterns,
                    'preprocessed': cleaned
                })

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding='post')
        pred = float(sql_model.predict(padded)[0][0])

        # Add these debug prints:
        print(f"Cleaned Query: {cleaned}")
        print(f"Model Prediction (raw): {pred}")

        # === Adjust prediction based on patterns ===
        if patterns:
            if pred < 0.9:
                pred = 0.95  # Boost confidence if any malicious pattern is found
        else:
            if 0.5 < pred < 0.9:
                pred = 1 - pred  # Reduce confidence if no patterns but model seems unsure

        # Final label decision
        if pred >= 0.9:
            label = "Injection Found"
        elif pred <= 0.1:
            label = "Legitimate Query"
        else:
            label = "Suspicious/Uncertain"


        return jsonify({
            'prediction': 1 if pred >= 0.9 else 0,  # 1 for malicious, 0 for safe
            'confidence': pred,
            'patterns': patterns,
            'preprocessed': cleaned
        })

    except Exception as e:
        print(f"[SQL Prediction Error] {e}")
        return jsonify({'error': str(e)}), 500

# === XSS Detection Endpoint ===
@app.route('/predict/xss', methods=['POST'])
def predict_xss():
    if xss_model is None or xss_vectorizer is None:
        # Instead of returning 503, provide a rule-based fallback
        data = request.get_json()
        payload = data.get('payload', '')

        if not payload.strip():
            return jsonify({'error': 'No payload provided'}), 400

        try:
            # Simple rule-based XSS detection as fallback
            clean_payload = payload.strip().lower()
            
            # XSS patterns to detect
            xss_patterns = [
                '<script', 'javascript:', 'onerror=', 'onload=', 'onclick=',
                'onmouseover=', '<iframe', 'alert(', 'eval(', 'document.cookie',
                'document.write', '<img', 'src=', 'vbscript:', '<object', '<embed'
            ]
            
            detected_patterns = []
            for pattern in xss_patterns:
                if pattern in clean_payload:
                    detected_patterns.append(pattern)
            
            if detected_patterns:
                pred = 1  # XSS detected
                confidence = min(0.7 + (len(detected_patterns) * 0.05), 0.95)
            else:
                pred = 0  # Safe
                confidence = 0.8
            
            label_map = {0: "Normal", 1: "XSS Attack"}

            return jsonify({
                "prediction": int(pred),
                "label": label_map[int(pred)],
                "confidence": float(confidence),
                "preprocessed": clean_payload,
                "note": "Using rule-based detection (ML model not available)"
            })

        except Exception as e:
            print(f"[XSS Prediction Error] {e}")
            return jsonify({'error': str(e)}), 500
    
    # Original ML-based code
    data = request.get_json()
    payload = data.get('payload', '')

    if not payload.strip():
        return jsonify({'error': 'No payload provided'}), 400

    try:
        clean_payload = payload.strip().lower()

        xss_signature_patterns = [
            r"<\s*script", r"javascript\s*:", r"vbscript\s*:",
            r"\bon\w+\s*=", r"<\s*(iframe|img|object|embed)",
            r"alert\s*\(", r"eval\s*\(", r"document\.(cookie|write)"
        ]

        has_xss_signature = any(
            re.search(pattern, clean_payload, flags=re.IGNORECASE)
            for pattern in xss_signature_patterns
        )

        if not has_xss_signature and not re.search(r"[<>=()'\";:/\\]", clean_payload):
            return jsonify({
                "prediction": 0,
                "label": "Normal",
                "confidence": 0.95,
                "preprocessed": clean_payload
            })

        vectorized = xss_vectorizer.transform([clean_payload])
        pred = xss_model.predict(vectorized)[0]
        prob = xss_model.predict_proba(vectorized)[0][pred]

        label_map = {0: "Normal", 1: "XSS Attack"}

        print("\n[DEBUG] XSS Payload:", clean_payload)
        print("[DEBUG] Vector Shape:", vectorized.shape)
        print("[DEBUG] Prediction:", pred)
        print("[DEBUG] Confidence:", prob)

        return jsonify({
            "prediction": int(pred),
            "label": label_map[int(pred)],
            "confidence": float(prob),
            "preprocessed": clean_payload
        })

    except Exception as e:
        print(f"[XSS Prediction Error] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)