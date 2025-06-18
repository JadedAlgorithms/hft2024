🛡️ InfoSafe

InfoSafe is an NLP-powered Chrome extension that helps users assess the reliability of news articles by analyzing their content. Using a combination of keyword extraction, semantic similarity, and contradiction detection, InfoSafe delivers a percentage-based trust score that reflects how trustworthy a given article is.
🚀 Features

    🔍 URL Analysis: Automatically fetches and analyzes the current tab's URL.

    🧠 NLP Models: Uses HuggingFace Transformers for sentence embeddings and contradiction checks.

    📊 Reliability Score: Combines semantic similarity, keyword matching, and contradiction detection to compute a trustworthiness percentage.

    🌐 Trusted Sources: Compares news articles against established media like Reuters, AP News, BBC, and FactCheck.org.

    🎨 Visual Feedback: Page background color changes from red to green based on reliability score.
🧠 How It Works

    Content Extraction: The extension grabs the full text from the article currently open in the active tab.

    Keyword Matching: Extracts key terms using NLTK to check for topic relevance.

    Semantic Similarity: Compares the article against trusted sources using Sentence Transformers to detect factual alignment.

    Contradiction Detection: Uses RoBERTa-large-MNLI to check for logical contradictions.

    Score Calculation: Outputs a reliability percentage, factoring in:

        Content similarity (50%)

        Keyword alignment (20%)

        Contradiction absence (30%)
    🛠️ Tech Stack

    Frontend: HTML, CSS, JS

    Backend: Python (Flask)

    NLP Models:

        sentence-transformers/all-MiniLM-L6-v2

        roberta-large-mnli

    Libraries:

        HuggingFace Transformers

        NLTK

        BeautifulSoup

        PyTorch

        Flask

        Pyngrok

🧪 Demo

    💡 Start the backend server before launching the Chrome extension.

    Clone this repo and run the backend Flask app:

    python infosafe_backend.py

    Start a tunnel (automatically done via pyngrok) and update the backend URL in:

        background.js

        manifest.json

    Load the extension into Chrome:

        Go to chrome://extensions

        Enable Developer Mode

        Click "Load unpacked" and select the folder

🧑‍💻 Team

    Johan – Project Lead & Backend NLP Developer

    @simonknowsstuff – Frontend & Extension Integration

    @AlvinGeorge-AG – UX Design & Data Flow

