# रेलवे OMR स्वतः जाँच प्रणाली

Hindi mobile-friendly web app: upload the official answer key and completed OMR sheet (PDF/JPG/PNG), then calculate marks. Files are processed in memory by the application and are not intentionally saved to disk.

## Deploy to Render
1. Upload this project folder to a GitHub repository.
2. In Render, select **New → Web Service** and connect the repository.
3. Choose **Docker** (the included `Dockerfile` installs Tesseract OCR and Poppler needed for scanned PDFs).
4. Deploy. Open the generated service URL on your phone. Health check: `/healthz`.

## Run locally with Docker
```bash
docker build -t railway-omr .
docker run --rm -p 8000:8000 railway-omr
```
Open `http://localhost:8000`.

## Run locally with Python
Install Tesseract OCR and Poppler on your computer first, then:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Important accuracy note
This tool uses OCR and image-based bubble detection. No OMR system can promise 100% accuracy for every scan/photo. Review any warnings, unclear marks, and the per-question result before relying on the final score. Use the official exam marking rules.

Maximum upload size: 25 MB per request; file loader supports up to 30 PDF pages.
