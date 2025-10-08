# CV/JD Analysis and Selection Tool

## Overview
A tool for extracting key skills from CVs and Job Descriptions (JDs) using NLP, and performing semantic matching to rank CVs. Built with Python, Spacy, pdfplumber, and Streamlit for a modular, scalable design.

## Features
- Automatic skill extraction from PDFs and text files.
- Semantic matching using Spacy embeddings.
- Customizable similarity threshold and skill dictionary.
- Streamlit UI for uploads, results display, and CSV export.
- Batch processing and caching for performance.
- Unit tests with pytest.

## Setup
1. Clone the repo: `git clone https://github.com/your-username/cv-jd-analysis-tool.git`
2. Create and activate venv: `python -m venv venv` then `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3. Install dependencies: `pip install -r requirements.txt`
4. Download Spacy model: `python -m spacy download en_core_web_md`

## Usage
- Run the app: `streamlit run app.py`
- Upload a JD (PDF/TXT) and multiple CVs (PDFs).
- Adjust settings in the sidebar (e.g., similarity threshold).
- Click "Perform Matching" to view ranked results, progress bars, and download CSV.

## Configuration
- Edit `config.yaml` for NLP model, file limits, and matching thresholds.
- Customize skills in `data/skills.json` with synonyms for better accuracy.

## Testing
- Run unit tests: `pytest tests/`
- End-to-end: Use sample files in `uploads/` and verify results in the UI.

## Optimization and Extensions
- **Performance**: Batch NLP and caching reduce processing time for large CV sets.
- **Extensions**: Add vector database (e.g., FAISS) for ultra-large datasets, multi-language support, or database integration (e.g., SQLite for storing results).
- **Deployment**: See below for Docker setup.

## Troubleshooting
- Check logs in `logs/` for errors.
- Ensure Python 3.12 and dependencies are installed.
- For issues with PDFs, verify `pdfplumber` handles your file formats.

## Contributors
- Adam Chebbi - Project Lead