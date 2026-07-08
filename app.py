"""
Agent IA — Assistant Santé Burkina Faso
VERSION 4 (TF-IDF + LLaMA 3.1 + 3 onglets + évaluation + mémoire)

LANCEMENT : streamlit run app.py
"""

import json
import math
import os
import re
import subprocess
import sys
from collections import Counter

import streamlit as st
from groq import Groq

# ─────────────────────────────────────────────
# AUTO-INGESTION AU DÉMARRAGE (pour Streamlit Cloud)
# ─────────────────────────────────────────────
def auto_ingest():
    """Lance ingest.py automatiquement si les fichiers JSON sont absents."""
    if not os.path.exists("./knowledge_base.json") or not os.path.exists("./tfidf_index.json"):
        with st.spinner(" Première initialisation — construction de la base TF-IDF..."):
            result = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f" Erreur lors de l'ingestion :\n{result.stderr}")
                st.stop()

auto_ingest()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
KB_FILE       = "./knowledge_base.json"
TFIDF_FILE    = "./tfidf_index.json"
GROQ_MODEL    = "llama-3.1-8b-instant"
MAX_SEGMENTS  = 5
MAX_HISTORY   = 8   # nombre de messages gardés en mémoire (nombre pair)

STOPWORDS = {
    "le","la","les","de","du","des","un","une","en","et","est","que","qui",
    "il","elle","ils","elles","nous","vous","on","se","ce","cet","cette","ces",
    "au","aux","par","pour","sur","sous","dans","avec","sans","ou","mais",
    "donc","ni","car","si","ne","pas","plus","très","aussi","bien","tout","tous",
    "leur","leurs","mon","ton","son","ma","ta","sa","notre","votre","être","avoir",
    "faire","dit","peut","doit","faut","comme","dont","quand","après","avant",
    "lors","même","autre","autres","puis","ainsi","lors","afin","dont",
}

QUESTIONS_TEST = [
    # --- In-domain : doivent trouver des passages pertinents ---
    {
        "question": "Quels sont les symptômes du paludisme ?",
        "mots_cles": ["fièvre", "frissons", "paludisme"],
        "domaine": "in-domain",
    },
    {
        "question": "Comment prévenir la dengue ?",
        "mots_cles": ["moustique", "gîtes", "dengue"],
        "domaine": "in-domain",
    },
    {
        "question": "Quelle alimentation pour un enfant de 2 ans ?",
        "mots_cles": ["protéines", "légumes", "lait"],
        "domaine": "in-domain",
    },
    {
        "question": "Où se trouve le CHU Yalgado Ouédraogo ?",
        "mots_cles": ["ouagadougou", "yalgado", "hôpital"],
        "domaine": "in-domain",
    },
    {
        "question": "Quelle est la différence entre dengue et paludisme ?",
        "mots_cles": ["fièvre", "articulaires", "moustique"],
        "domaine": "in-domain",
    },
    # --- Hors-domaine : le score cosinus doit être bas ---
    {
        "question": "Quel est le prix du kilo d'or au Burkina Faso ?",
        "mots_cles": [],
        "domaine": "hors-domaine",
        "seuil_max": 0.15,
    },
    {
        "question": "Qui est le président du Burkina Faso en 2024 ?",
        "mots_cles": [],
        "domaine": "hors-domaine",
        "seuil_max": 0.15,
    },
    {
        "question": "Quelles sont les règles du football ?",
        "mots_cles": [],
        "domaine": "hors-domaine",
        "seuil_max": 0.15,
    },
]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant Santé Burkina Faso",
    layout="wide",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #006400, #228B22, #2E8B57);
    color: white; padding: 22px 28px; border-radius: 14px;
    text-align: center; margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,100,0,0.25);
}
.main-header h1 { margin: 0 0 6px 0; font-size: 1.9em; }
.main-header p  { margin: 0; font-size: 1em; opacity: 0.9; }

.warning-box {
    background: #fff3e0; border-left: 4px solid #ff9800;
    padding: 10px 14px; border-radius: 6px; margin-bottom: 14px;
    font-size: 0.92em;
}
.score-box {
    background: #e8f5e9; border-left: 4px solid #388e3c;
    padding: 6px 12px; border-radius: 4px; margin-top: 8px;
    font-size: 0.88em;
}
.score-box.faible {
    background: #fff8e1; border-left-color: #f9a825;
}
.eval-pass { color: #2e7d32; font-weight: bold; }
.eval-fail { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Drapeau Burkina Faso (SVG inline)
    st.markdown("""
    <div style="text-align:center;margin-bottom:10px;">
      <svg width="80" height="53" viewBox="0 0 3 2" xmlns="http://www.w3.org/2000/svg">
        <rect width="3" height="1" fill="#EF2B2D"/>
        <rect y="1" width="3" height="1" fill="#009A00"/>
        <polygon points="1.5,0.55 1.65,1.0 1.1,0.7 1.9,0.7 1.35,1.0"
                 fill="#FCD116"/>
      </svg>
      <div style="font-size:0.8em;color:#555;">Burkina Faso</div>
    </div>
    """, unsafe_allow_html=True)

    st.header(" Configuration")

    # Priorité : secrets Streamlit Cloud > variable d'env > saisie manuelle
    groq_api_key = (
        st.secrets.get("GROQ_API_KEY", "")
        or os.environ.get("GROQ_API_KEY", "")
    )
    if not groq_api_key:
        groq_api_key = st.text_input(
            "Clé API Groq",
            type="password",
            placeholder="gsk_...",
            help="Gratuit sur https://console.groq.com/keys",
        )
        st.caption(" [Créer un compte Groq gratuit](https://console.groq.com)")
    else:
        st.success(" Clé API chargée")

    st.divider()
    temperature = st.slider("Créativité (température)", 0.0, 1.0, 0.2, 0.1,
                            help="0 = réponses plus sûres, 1 = plus créatives")

    st.divider()
    if st.button(" Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = {"questions": 0, "hallucinations": 0}
        st.rerun()

    st.divider()
    st.markdown("** Questions rapides :**")
    exemples = [
        "Symptômes du paludisme ?",
        "Prévenir la dengue ?",
        "Alimentation enfant 2 ans ?",
        "CHU Yalgado : où est-il ?",
        "Différence dengue/paludisme ?",
        "Que faire pour le choléra ?",
        "Vaccins pour les enfants ?",
    ]
    for q in exemples:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q


# ─────────────────────────────────────────────
# CHARGEMENT (base + index TF-IDF)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=" Chargement de la base de connaissances...")
def charger_donnees():
    if not os.path.exists(KB_FILE) or not os.path.exists(TFIDF_FILE):
        return [], {}, []

    with open(KB_FILE, "r", encoding="utf-8") as f:
        segments = json.load(f)

    with open(TFIDF_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    return segments, index["idf"], index["vecteurs"]


# ─────────────────────────────────────────────
# PIPELINE TF-IDF
# ─────────────────────────────────────────────
def tokeniser(texte: str) -> list:
    tokens = re.findall(r"[a-zàâäéèêëîïôùûüç]{3,}", texte.lower())
    return [t for t in tokens if t not in STOPWORDS]


def norme(vecteur: dict) -> float:
    return math.sqrt(sum(v * v for v in vecteur.values())) or 1.0


def cosine_sim(v1: dict, v2: dict) -> float:
    communs = set(v1.keys()) & set(v2.keys())
    if not communs:
        return 0.0
    produit = sum(v1[t] * v2[t] for t in communs)
    return produit / (norme(v1) * norme(v2))


def vecteur_requete(tokens: list, idf: dict) -> dict:
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (freq / total) * idf.get(t, 1.0) for t, freq in tf.items()}


def rechercher(question: str, segments: list, idf: dict,
               vecteurs: list, k: int = MAX_SEGMENTS) -> list:
    tokens_q = tokeniser(question)
    if not tokens_q:
        return segments[:k], [0.0] * k

    vq = vecteur_requete(tokens_q, idf)
    scores = [cosine_sim(vq, vd) for vd in vecteurs]

    indices_tries = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_k = indices_tries[:k]

    return [segments[i] for i in top_k], [scores[i] for i in top_k]


# ─────────────────────────────────────────────
# DÉTECTION D'HALLUCINATION
# ─────────────────────────────────────────────
MARQUEURS_REFUS = [
    "je ne dispose pas", "cette information n'est pas",
    "hors de ma base", "consultez un professionnel",
    "je n'ai pas d'information", "pas dans ma base",
]

def detecter_hallucination(reponse: str) -> bool:
    r = reponse.lower()
    return any(m in r for m in MARQUEURS_REFUS)


# ─────────────────────────────────────────────
# GÉNÉRATION (RAG + mémoire)
# ─────────────────────────────────────────────
def generer_reponse(question: str, passages: list, historique: list,
                    api_key: str, temp: float) -> str:

    contexte = "\n\n---\n\n".join(p["texte"] for p in passages)
    sources   = list({p["source"] for p in passages})

    system_prompt = f"""Tu es un assistant médical et de prévention sanitaire officiel pour le Burkina Faso.
Tu réponds uniquement aux questions de santé : paludisme, dengue, nutrition, cholera, méningite, santé maternelle, vaccination, pharmacies, centres de santé.

RÈGLES ABSOLUES :
1. Utilise UNIQUEMENT les informations du CONTEXTE ci-dessous.
2. Si une information est absente du contexte, réponds : "Je ne dispose pas de cette information dans ma base de connaissances. Consultez un professionnel de santé ou appelez le 112."
3. Ne fabrique JAMAIS de dosage ou traitement spécifique non mentionné dans le contexte.
4. Pour toute urgence, mentionne systématiquement le 112 (SAMU Burkina Faso).
5. Termine chaque réponse par les sources documentaires utilisées.
6. Réponds toujours en français, de façon claire et accessible.

CONTEXTE (base de connaissances santé Burkina Faso) :
{contexte}

Sources disponibles : {', '.join(sources)}"""

    # Construction des messages avec historique
    messages = [{"role": "system", "content": system_prompt}]

    # Ajouter l'historique (fenêtre glissante)
    hist = historique[-MAX_HISTORY:]
    for msg in hist:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temp,
        max_tokens=1200,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
segments, idf, vecteurs = charger_donnees()

if not segments:
    st.error("""
     **Base de connaissances introuvable.**
    Lancez d'abord dans votre terminal (depuis le dossier du projet) :
    ```
    python ingest.py
    ```
    """)
    st.stop()

# ─────────────────────────────────────────────
# ÉTAT DE SESSION
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Bonjour !  Je suis votre assistant santé pour le Burkina Faso.\n\n"
            "Je peux vous aider sur :\n"
            "-  **Paludisme & Dengue** : symptômes, prévention, traitement\n"
            "-  **Nutrition** : alimentation saine avec produits locaux\n"
            "-  **Choléra, Méningite, Typhoïde** : prévention & soins\n"
            "-  **Santé maternelle & Vaccination** : CPN, calendrier PEV\n"
            "-  **Pharmacies & CHU** : adresses et contacts d'urgence\n\n"
            "Utilisez le bouton **Évaluation** pour tester mes performances.\n"
            "En cas d'urgence, appelez le **112**."
        ),
    }]

if "stats" not in st.session_state:
    st.session_state.stats = {"questions": 0, "hallucinations": 0}


# ─────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1> Assistant Santé Burkina Faso</h1>
    <p>Paludisme · Dengue · Nutrition · Choléra · Santé Maternelle · Vaccination · Pharmacies</p>
</div>
<div class="warning-box">
     <b>Avertissement médical</b> : Ces informations sont générales.
    Elles ne remplacent pas une consultation médicale.
    En cas d'urgence : <b>112 (SAMU)</b>.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════════════════════════
tab_chat, tab_eval, tab_arch = st.tabs([
    " Consultation",
    " Évaluation (Étape 4)",
    " Architecture",
])


# ───────────────────────────────────────────────────────────────────
# ONGLET 1 : CONSULTATION
# ───────────────────────────────────────────────────────────────────
with tab_chat:
    col_info, col_stat = st.columns([3, 1])
    with col_info:
        st.caption(f" {len(segments)} segments indexés · Modèle : {GROQ_MODEL}")
    with col_stat:
        q = st.session_state.stats["questions"]
        h = st.session_state.stats["hallucinations"]
        if q > 0:
            st.caption(f" {q} question(s) ·  {h} hors-domaine")

    st.divider()

    # Affichage de l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("score") is not None:
                score_pct = msg["score"] * 100
                css_class = "score-box" if score_pct >= 10 else "score-box faible"
                st.markdown(
                    f'<div class="{css_class}"> Score de pertinence RAG : {score_pct:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            if msg.get("passages"):
                with st.expander(" Passages RAG utilisés", expanded=False):
                    for i, p in enumerate(msg["passages"], 1):
                        st.markdown(f"**Source {i} — {p['source']}**")
                        st.markdown(f"_{p['texte'][:300]}..._")
                        st.divider()

            if msg.get("sources"):
                with st.expander(" Sources documentaires", expanded=False):
                    for s in msg["sources"]:
                        st.caption(f"• {s}")

    # Saisie
    user_input = (
        st.session_state.pop("pending_question", None)
        or st.chat_input("Posez votre question de santé...")
    )

    if user_input:
        if not groq_api_key:
            st.warning(" Entrez d'abord votre clé API Groq dans la barre latérale.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner(" Recherche TF-IDF + génération LLaMA 3.1..."):
                    try:
                        # Pipeline RAG
                        passages_trouves, scores_cosinus = rechercher(
                            user_input, segments, idf, vecteurs
                        )
                        score_max = scores_cosinus[0] if scores_cosinus else 0.0

                        # Historique pour la mémoire
                        historique_prec = [
                            m for m in st.session_state.messages
                            if m["role"] in ("user", "assistant")
                        ][:-1]  # exclure le message courant

                        reponse = generer_reponse(
                            user_input, passages_trouves,
                            historique_prec, groq_api_key, temperature
                        )

                        est_hallucination = detecter_hallucination(reponse)

                        # Stats
                        st.session_state.stats["questions"] += 1
                        if est_hallucination:
                            st.session_state.stats["hallucinations"] += 1

                        # Affichage
                        st.markdown(reponse)

                        if not est_hallucination:
                            score_pct = score_max * 100
                            css = "score-box" if score_pct >= 10 else "score-box faible"
                            st.markdown(
                                f'<div class="{css}"> Score de pertinence RAG : {score_pct:.1f}%</div>',
                                unsafe_allow_html=True,
                            )

                        with st.expander(" Passages RAG utilisés", expanded=False):
                            for i, p in enumerate(passages_trouves, 1):
                                st.markdown(f"**Source {i} — {p['source']}** (score : {scores_cosinus[i-1]:.3f})")
                                st.markdown(f"_{p['texte'][:350]}..._")
                                st.divider()

                        sources = list({p["source"] for p in passages_trouves})
                        with st.expander(" Sources documentaires", expanded=False):
                            for s in sources:
                                st.caption(f"• {s}")

                        # Sauvegarder dans l'historique
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reponse,
                            "score": score_max if not est_hallucination else None,
                            "passages": passages_trouves,
                            "sources": sources,
                        })

                    except Exception as e:
                        st.error(f" Erreur : {e}")


# ───────────────────────────────────────────────────────────────────
# ONGLET 2 : ÉVALUATION
# ───────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader(" Évaluation automatique du système RAG")
    st.markdown("""
    Ce module évalue deux dimensions de la robustesse :
    - **Precision@k** : les passages récupérés contiennent-ils les mots-clés attendus ?
    - **Taux de refus** : les questions hors-domaine ont-elles un faible score cosinus (< 15%) ?
    """)

    if not idf:
        st.warning(" Index TF-IDF non chargé. Lancez d'abord `python ingest.py`.")
    else:
        if st.button(" Lancer l'évaluation complète (8 tests)", type="primary"):
            resultats = []
            progress_bar = st.progress(0, text="Évaluation en cours...")

            for i, test in enumerate(QUESTIONS_TEST):
                progress_bar.progress((i + 1) / len(QUESTIONS_TEST),
                                      text=f"Test {i+1}/{len(QUESTIONS_TEST)} : {test['question'][:40]}...")

                passages_r, scores_r = rechercher(
                    test["question"], segments, idf, vecteurs
                )
                score_max = scores_r[0] if scores_r else 0.0
                texte_concat = " ".join(p["texte"].lower() for p in passages_r)

                if test["domaine"] == "in-domain":
                    mots_trouves = [m for m in test["mots_cles"] if m.lower() in texte_concat]
                    succes = len(mots_trouves) >= max(1, len(test["mots_cles"]) // 2)
                    detail = f"{len(mots_trouves)}/{len(test['mots_cles'])} mots-clés trouvés · score cosinus : {score_max:.3f}"
                else:
                    seuil = test.get("seuil_max", 0.15)
                    succes = score_max < seuil
                    detail = f"Score cosinus : {score_max:.3f} (seuil : < {seuil})"

                resultats.append({
                    "question": test["question"],
                    "domaine": test["domaine"],
                    "succes": succes,
                    "detail": detail,
                    "score": score_max,
                })

            progress_bar.empty()

            # Résultats
            nb_succes = sum(r["succes"] for r in resultats)
            nb_total  = len(resultats)
            pct       = nb_succes / nb_total * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Tests réussis", f"{nb_succes}/{nb_total}")
            col2.metric("Score global", f"{pct:.0f}%")
            col3.metric("Taux d'échec", f"{nb_total - nb_succes}/{nb_total}")

            if pct >= 80:
                st.success(f" Système validé ({pct:.0f}% de réussite)")
            elif pct >= 60:
                st.warning(f" Résultats partiels ({pct:.0f}%)")
            else:
                st.error(f" Résultats insuffisants ({pct:.0f}%)")

            st.divider()

            # Tableau des résultats
            for r in resultats:
                status = "PASS" if r["succes"] else "FAIL"
                dom_icon = "Dans le domaine" if r["domaine"] == "in-domain" else "Hors domaine"
                css_class = "eval-pass" if r["succes"] else "eval-fail"
                with st.expander(f"{dom_icon} {r['question']}", expanded=False):
                    st.markdown(f'<span class="{css_class}">{status}</span>', unsafe_allow_html=True)
                    st.caption(f"Type : {r['domaine']} | {r['detail']}")

        st.divider()
        st.subheader(" Bilan de la session actuelle")
        q = st.session_state.stats["questions"]
        h = st.session_state.stats["hallucinations"]
        if q > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Questions posées", q)
            col2.metric("Réponses hors-domaine", h)
            col3.metric("Taux de réponse in-domain", f"{(q-h)/q*100:.0f}%")
        else:
            st.info("Aucune question posée dans cette session. Allez dans l'onglet Consultation.")


# ───────────────────────────────────────────────────────────────────
# ONGLET 3 : ARCHITECTURE
# ───────────────────────────────────────────────────────────────────
with tab_arch:
    st.subheader(" Architecture du système RAG")

    st.markdown("### Flux de données")
    st.code("""
┌──────────────────────────────────────────────────────────────┐
│                    PHASE D'INGESTION                         │
│                      ingest.py                               │
│                                                              │
│  Fichiers .txt ──► Chunking (500 car.) ──► Tokenisation      │
│   data/*.txt        + overlap 80 car.      + stopwords FR    │
│                                  │                           │
│                           Calcul IDF ──► knowledge_base.json │
│                        (Laplace smooth)   tfidf_index.json   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  PHASE DE CONSULTATION                       │
│                        app.py                                │
│                                                              │
│  Question ──► Tokenisation ──► Vecteur TF-IDF (requête)      │
│                                       │                      │
│                          Similarité cosinus avec index       │
│                                       │                      │
│                          Top-5 segments les + pertinents     │
│                                       │                      │
│   Historique (4 échanges) ──► PROMPT ──► Groq / LLaMA 3.1    │
│                                       │                      │
│              Réponse + Score + Sources + Anti-hallucination  │
└──────────────────────────────────────────────────────────────┘
    """, language="text")

    st.markdown("### Stack technologique")
    tech_data = {
        "Composant": ["Retrieval", "LLM", "Interface", "Stockage", "Mémoire",
                      "Anti-hallucination", "Évaluation"],
        "Technologie": ["TF-IDF + cosinus (Python natif)", "LLaMA 3.1-8b-instant (Groq API)",
                        "Streamlit 1.35+", "JSON (knowledge_base + index TF-IDF)",
                        "Session state (4 derniers échanges)", "Prompt engineering strict",
                        "Precision@k + seuil cosinus hors-domaine"],
        "Avantage": ["0 dépendance ML, déterministe", "Gratuit, < 1 sec, 131k tokens",
                     "Web app déployable", "Léger, lisible, portable",
                     "Contexte conversationnel", "Limite les réponses inventées",
                     "Intégré à l'interface (Étape 4)"],
    }

    col_h1, col_h2, col_h3 = st.columns([2, 3, 3])
    col_h1.markdown("**Composant**")
    col_h2.markdown("**Technologie**")
    col_h3.markdown("**Avantage**")
    st.divider()
    for c, t, a in zip(tech_data["Composant"], tech_data["Technologie"], tech_data["Avantage"]):
        c1, c2, c3 = st.columns([2, 3, 3])
        c1.write(c)
        c2.write(t)
        c3.write(a)

    st.markdown("### Formules TF-IDF implémentées")
    st.latex(r"TF(t,d) = \frac{f_{t,d}}{|d|}")
    st.latex(r"IDF(t) = \log\frac{N+1}{df(t)+1} + 1 \quad \text{(Laplace smoothing)}")
    st.latex(r"TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)")
    st.latex(r"\text{Score}(q,d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \cdot \|\vec{d}\|}")

    st.markdown("### Base de connaissances")
    kb_info = [
        ("paludisme", "~8 segments"),
        ("dengue", "~8 segments"),
        ("nutrition", "~12 segments"),
        ("pharmacies_centres_sante", "~11 segments"),
        ("cholera_hygiene", "~14 segments"),
        ("sante_maternelle_vaccination", "~16 segments"),
    ]
    for nom, nb in kb_info:
        st.markdown(f"- `data/{nom}.txt` → {nb}")
    st.markdown(f"\n**Total indexé dans cette session : {len(segments)} segments**")
