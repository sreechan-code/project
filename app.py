from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os
import random
import string
import re
import smtplib
from email.mime.text import MIMEText
import bcrypt
from smtplib import SMTPAuthenticationError, SMTPServerDisconnected, SMTPConnectError
from datetime import datetime
import pytz

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# SQLAlchemy configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_EMAIL = 'sreechandharilal123@gmail.com'
SMTP_PASSWORD = 'kbvkytxloadavebk'

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False)
    otp = db.Column(db.String(6), nullable=True)
    activities = db.relationship('UserActivity', backref='user', lazy=True)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

# UserActivity model
class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    event = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(pytz.timezone('Asia/Kolkata')))

# Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(pytz.timezone('Asia/Kolkata')))

# Model training and serialization (unchanged)
def load_model():
    try:
        csv_path = 'breast-cancer-wisconsin-data.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
        
        data = pd.read_csv(csv_path)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

        selected_features = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'
        ]
        
        X = data[selected_features]
        y = data['diagnosis']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("Model and scaler saved successfully.")
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise

def get_model_and_scaler():
    try:
        if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
            raise FileNotFoundError("Model or scaler file not found.")
        
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"Error in get_model_and_scaler: {str(e)}")
        raise

# Updated email validation function
def is_valid_email(email):
    if email == 'admin@example.com':
        return True
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email, re.IGNORECASE) is not None

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_otp(email, otp):
    try:
        if email == 'admin@example.com':
            print(f"OTP for {email}: {otp}")
            return True
        else:
            msg = MIMEText(f"Your password reset OTP is: {otp}")
            msg['Subject'] = 'Breast Cancer Prediction - Password Reset'
            msg['From'] = SMTP_EMAIL
            msg['To'] = email

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.set_debuglevel(1)
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASSWORD)
                server.sendmail(SMTP_EMAIL, email, msg.as_string())
            print(f"Email sent to {email}")
            return True
    except SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: Invalid email or password. Ensure SMTP_EMAIL and SMTP_PASSWORD are correct. {str(e)}")
        return False
    except SMTPConnectError as e:
        print(f"SMTP Connect Error: Could not connect to {SMTP_SERVER}:{SMTP_PORT}. Check network or firewall. {str(e)}")
        return False
    except SMTPServerDisconnected as e:
        print(f"SMTP Server Disconnected: Server closed connection unexpectedly. {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error sending OTP to {email}: {str(e)}")
        return False

# Model initialization (unchanged)
try:
    load_model()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    exit(1)

# Routes
@app.route('/')
def index():
    print(f"Index: logged_in={session.get('logged_in')}, is_admin={session.get('is_admin')}")
    if session.get('logged_in'):
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']

        if not is_valid_email(email):
            error = 'Invalid email address.'
            return render_template('register.html', error=error)

        if len(name) < 3:
            error = 'Name must be at least 3 characters long.'
            return render_template('register.html', error=error)

        if len(password) < 6:
            error = 'Password must be at least 6 characters long.'
            return render_template('register.html', error=error)

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            error = 'Email already exists.'
            return render_template('register.html', error=error)

        existing_name = User.query.filter_by(name=name).first()
        if existing_name:
            error = 'Name already exists.'
            return render_template('register.html', error=error)

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        otp = generate_otp()

        new_user = User(
            email=email,
            name=name,
            password=hashed_password.decode('utf-8'),
            is_admin=(email == 'admin@example.com'),
            otp=otp
        )
        db.session.add(new_user)
        db.session.commit()

        activity = UserActivity(user_id=new_user.id, event=f"User {name} registered with email {email}")
        db.session.add(activity)
        db.session.commit()

        if send_otp(email, otp):
            session['pending_verification_email'] = email
            return redirect(url_for('verify'))
        else:
            db.session.delete(new_user)
            db.session.commit()
            error = 'Failed to send OTP.'
            return render_template('register.html', error=error)
    
    return render_template('register.html', error=error)

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    error = None
    email = session.get('pending_verification_email')
    if not email:
        return redirect(url_for('register'))

    if request.method == 'POST':
        otp = request.form['otp']
        user = User.query.filter_by(email=email).first()

        if user and user.otp == otp:
            user.is_verified = True
            user.otp = None
            db.session.commit()
            session.pop('pending_verification_email', None)
            return redirect(url_for('login'))
        else:
            error = 'Invalid OTP.'

    return render_template('verify.html', error=error, email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        user = User.query.filter_by(name=name).first()
        if user and user.is_verified and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            session['logged_in'] = True
            session['is_admin'] = user.is_admin
            session['name'] = user.name
            session['user_id'] = user.id
            print(f"Login: name={name}, logged_in={session['logged_in']}, is_admin={session['is_admin']}, user_id={session['user_id']}")
            activity = UserActivity(user_id=user.id, event=f"User {name} logged in")
            db.session.add(activity)
            db.session.commit()
            return redirect(url_for('home'))
        else:
            error = 'Invalid name, password, or unverified account.'
    
    logged_in = session.get('logged_in', False)
    is_admin = session.get('is_admin', False)
    print(f"Rendering login: logged_in={logged_in}, is_admin={is_admin}")
    return render_template('login.html', error=error, logged_in=logged_in, is_admin=is_admin)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        
        if not is_valid_email(email):
            error = 'Invalid email address.'
            return render_template('forgot_password.html', error=error)
        
        user = User.query.filter_by(email=email).first()
        if not user:
            error = 'No account found with this email.'
            return render_template('forgot_password.html', error=error)
        
        if not user.is_verified:
            error = 'Account is not verified.'
            return render_template('forgot_password.html', error=error)
        
        otp = generate_otp()
        user.otp = otp
        db.session.commit()
        
        if send_otp(email, otp):
            session['reset_password_email'] = email
            return redirect(url_for('forgot_otp'))
        else:
            error = 'Failed to send OTP.'
            return render_template('forgot_password.html', error=error)
    
    return render_template('forgot_password.html', error=error)

@app.route('/forgot_otp', methods=['GET', 'POST'])
def forgot_otp():
    error = None
    email = session.get('reset_password_email')
    if not email:
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        otp = request.form['otp']
        user = User.query.filter_by(email=email).first()
        
        if user and user.otp == otp:
            user.otp = None
            db.session.commit()
            return redirect(url_for('reset_password'))
        else:
            error = 'Invalid OTP.'
    
    return render_template('forgot_otp.html', error=error, email=email)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    error = None
    email = session.get('reset_password_email')
    if not email:
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if len(password) < 6:
            error = 'Password must be at least 6 characters long.'
            return render_template('reset_password.html', error=error)
        
        if password != confirm_password:
            error = 'Passwords do not match.'
            return render_template('reset_password.html', error=error)
        
        user = User.query.filter_by(email=email).first()
        if user:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user.password = hashed_password.decode('utf-8')
            activity = UserActivity(user_id=user.id, event=f"User {user.name} reset password")
            db.session.add(activity)
            db.session.commit()
            session.pop('reset_password_email', None)
            return redirect(url_for('login'))
        else:
            error = 'User not found.'
            return render_template('reset_password.html', error=error)
    
    return render_template('reset_password.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    is_admin = session.get('is_admin', False)
    current_time = datetime.now(pytz.timezone('Asia/Kolkata')).hour
    greeting_base = 'Good Morning' if current_time < 12 else 'Good Afternoon' if current_time < 17 else 'Good Evening'
    name = session.get('name', 'Guest')
    full_greeting = f"{greeting_base}, {name}!"  # Full greeting string
    print(f"Home: logged_in={session.get('logged_in')}, is_admin={is_admin}, time={current_time}, greeting={full_greeting}")
    return render_template('index.html', is_admin=is_admin, greeting=full_greeting)

@app.route('/predict_form')
def predict_form():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    user_id = session.get('user_id')
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).limit(5).all()
    return render_template('predict.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        features = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['concavity_mean']),
            float(request.form['concave_points_mean'])
        ]

        model, scaler = get_model_and_scaler()
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = 'Malignant' if prediction == 1 else 'Benign'
        probability_percent = probability * 100

        # Store prediction
        user_id = session.get('user_id')
        new_prediction = Prediction(
            user_id=user_id,
            result=result,
            probability=probability_percent,
            timestamp=datetime.now(pytz.timezone('Asia/Kolkata'))
        )
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('result.html', result=result, probability=probability_percent)
    
    except ValueError:
        return "Error: Please ensure all inputs are valid numbers.", 400
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/admin')
def admin():
    print(f"Admin access: logged_in={session.get('logged_in')}, is_admin={session.get('is_admin')}")
    if not session.get('logged_in') or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    users = User.query.all()
    activities = UserActivity.query.order_by(UserActivity.timestamp.desc()).all()
    return render_template('admin.html', users=users, activities=activities)

@app.route('/user_details/<int:user_id>')
def user_details(user_id):
    if not session.get('logged_in') or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    user = User.query.get_or_404(user_id)
    activities = UserActivity.query.filter_by(user_id=user_id).order_by(UserActivity.timestamp.desc()).all()
    return render_template('user_details.html', user=user, activities=activities)

@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    if not session.get('logged_in') or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    user = User.query.get(user_id)
    if user:
        UserActivity.query.filter_by(user_id=user_id).delete()
        Prediction.query.filter_by(user_id=user_id).delete()
        db.session.delete(user)
        db.session.commit()
    
    return redirect(url_for('admin'))

if __name__ == '__main__':
    with app.app_context():
        db.drop_all()
        db.create_all()
        if not User.query.filter_by(email='admin@example.com').first():
            hashed_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            admin_user = User(
                email='admin@example.com',
                name='Admin User',
                password=hashed_password.decode('utf-8'),
                is_admin=True,
                is_verified=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: email=admin@example.com, name=Admin User, is_admin=True")
    app.run(debug=True, port=5000)