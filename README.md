# PhishGuard – Phishing Website Detection

## Project Overview
PhishGuard is a web-based project that predicts whether a website URL may be phishing or legitimate using Machine Learning. It provides a simple interface where users can enter a URL and view the prediction.

## Objectives
- Identify suspicious URLs using URL-based features.
- Apply a Machine Learning classification model.
- Provide a simple, user-friendly web interface.
- Understand the use of Machine Learning in cybersecurity.

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas and NumPy
- HTML, CSS and JavaScript
- Joblib

## Machine Learning Model
The project uses a Random Forest Classifier trained on URL-based features, including URL length, IP address usage, URL shortening services, and subdomain information.

The URL-only model achieved approximately **72.5% accuracy** on its evaluation dataset.

## Dataset
The project uses the UCI Phishing Websites dataset for model training and evaluation.

## How to Run the Project
1. Open the project folder in VS Code.
2. Activate the virtual environment in the terminal:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Start the Flask application:
   ```powershell
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.
5. Enter a URL and click the analysis button.

## Limitations
- Predictions are based on URL features and may be incorrect.
- The model does not verify a website's actual content or safety.
- The project is an educational prototype and should not be used as the sole basis for security decisions.

## Future Scope
- Improve model accuracy using additional features.
- Evaluate the model on more diverse datasets.
- Add more security checks and improve the user interface.

## Disclaimer
PhishGuard is developed for educational purposes. A legitimate prediction does not guarantee that a website is safe.
