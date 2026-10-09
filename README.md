# Comment Sentiment

A small, dependency-free HTML and JavaScript app. It trains a Naive Bayes sentiment model in your browser from `train_data.csv`; label `0` means negative, label `2` means positive, and neutral examples (label `1`) are skipped.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m http.server 8000
```

Then open <http://localhost:8000>. Keep `train_data.csv`, `index.html`, `app.js`, and `styles.css` together in this folder. No virtual environment or Python packages are required; Python's built-in static file server only serves the page and CSV to your browser.
# ML-Based-Sentiment-Analysis-System
