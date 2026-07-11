# Assistant Santé Burkina Faso — Agent IA RAG

**Projet Data Science — Master 1 IFOAD/UJKZ — 2026**  
Option 3 : Agent d'Orientation Médicale & Prévention Sanitaire

[![Streamlit App](https://assistant-sante-burkinabf-ds5cavaptcy9zu4g6pbkgk.streamlit.app/)]

---

## Description

Agent conversationnel basé sur l'architecture **RAG (Retrieval-Augmented Generation)** pour fournir des informations médicales fiables aux citoyens burkinabè. Le système combine :

- **TF-IDF** (pure Python, sans ML) pour la recherche de passages pertinents
- **LLaMA 3.1-8b-instant** via l'API Groq pour la génération de réponses
- **Streamlit** pour l'interface web

## Thèmes couverts

- Paludisme (symptômes, prévention, traitement)
- Dengue (symptômes, prévention)
- Nutrition (alimentation locale burkinabè)
- Choléra, Méningite, Typhoïde
- Santé maternelle & Vaccination (PEV)
- Pharmacies & CHU (annuaire + contacts urgence)

## Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/laminesawadogo/assistant-sante-burkina_bf.git
cd assistant-sante-burkina

# 2. Installer les dépendances (seulement 2 packages !)
pip install -r requirements.txt

# 3. Construire l'index TF-IDF
python ingest.py

# 4. Lancer l'application
streamlit run app.py
```

## Déploiement Streamlit Cloud

1. Forker ce dépôt sur GitHub
2. Aller sur [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://votre-app.streamlit.app)
3. Connecter votre compte GitHub
4. Sélectionner ce dépôt → `app.py`
5. Dans **Settings → Secrets**, ajouter :
   ```
   GROQ_API_KEY = "gsk_votre_cle_ici"
   ```
6. Cliquer **Deploy** → URL publique générée automatiquement

## Architecture

```
Fichiers .txt → ingest.py (TF-IDF) → JSON index
                                          ↓
Question → Tokenisation → Cosinus → Top-5 passages → LLaMA 3.1 → Réponse
```

## Dépendances

```
streamlit>=1.35.0
groq>=0.9.0
```

Aucune bibliothèque ML (pas de PyTorch, pas de CUDA requis).

## Avertissement

> Cet assistant fournit des informations générales de santé publique.  
> Il ne remplace pas une consultation médicale.  
> **Urgences : 112 (SAMU Burkina Faso)**
