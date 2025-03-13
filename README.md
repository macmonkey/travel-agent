# Reiseplan-Agent

Ein KI-Agent zur Generierung personalisierter Reisepläne basierend auf Google Gemini und RAG-Technologie.

## Übersicht

Der Reiseplan-Agent nutzt:
- Google Gemini als KI-Modell
- Chroma DB als Vektordatenbank
- RAG-Technologie (Retrieval-Augmented Generation)
- Verarbeitung von PDF und DOCX Dokumenten

## Installation

1. Repository klonen oder Dateien herunterladen

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Gemini API-Key in `.env` Datei eintragen:
   ```
   GEMINI_API_KEY=dein-api-key-hier
   ```

## Nutzung

1. Reisedokumente (PDFs, DOCs) in den Ordner `data/documents` legen

2. Programm starten:
   ```bash
   python main.py
   ```

3. Reisewunsch eingeben und den Anweisungen folgen

## Funktionen

- Verarbeitung von PDFs und DOC/DOCX-Dokumenten
- Semantische Suche in der Vektordatenbank
- Generierung eines Reiseplan-Entwurfs
- Feedback-Integration
- Erstellung eines detaillierten Reiseplans

## Projektstruktur

- `main.py`: Haupteinstiegspunkt
- `config.py`: Konfigurationsparameter
- `document_processor.py`: Dokumentverarbeitung
- `rag_database.py`: Vektordatenbank-Verwaltung
- `agent.py`: KI-Agent mit Gemini-Integration
- `prompts.py`: Prompt-Templates
- `utils.py`: Hilfsfunktionen

## Erste Schritte

1. Stellen Sie sicher, dass Sie einen Gemini API-Key haben und diesen in die `.env` Datei eingetragen haben
2. Legen Sie mindestens ein Reisedokument (PDF oder DOCX) in den Ordner `data/documents`
3. Starten Sie das Programm und folgen Sie den Anweisungen