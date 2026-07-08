"""
Script d'ingestion — VERSION 4 (TF-IDF, pure Python, sans ML)
Lit les .txt du dossier data/ → découpage intelligent → index TF-IDF JSON

LANCEMENT : python ingest.py
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

DATA_DIR    = "./data"
KB_FILE     = "./knowledge_base.json"
TFIDF_FILE  = "./tfidf_index.json"
CHUNK_SIZE  = 500
CHUNK_OVER  = 80
MIN_CHUNK   = 60

# ─── Stopwords français ───────────────────────────────────────────────────────
STOPWORDS = {
    "le","la","les","de","du","des","un","une","en","et","est","que","qui",
    "il","elle","ils","elles","nous","vous","on","se","ce","cet","cette","ces",
    "au","aux","par","pour","sur","sous","dans","avec","sans","ou","mais",
    "donc","ni","car","si","ne","pas","plus","très","aussi","bien","tout","tous",
    "leur","leurs","mon","ton","son","ma","ta","sa","notre","votre","être","avoir",
    "faire","dit","peut","doit","faut","comme","dont","quand","après","avant",
    "lors","même","autre","autres","puis","ainsi","lors","lors","afin","dont",
}


# ─── Tokenisation ────────────────────────────────────────────────────────────
def tokeniser(texte: str) -> list:
    tokens = re.findall(r"[a-zàâäéèêëîïôùûüç]{3,}", texte.lower())
    return [t for t in tokens if t not in STOPWORDS]


# ─── Découpage intelligent ────────────────────────────────────────────────────
def decouper_texte(texte: str, source: str) -> list:
    segments = []
    debut = 0
    idx = 0

    while debut < len(texte):
        fin = min(debut + CHUNK_SIZE, len(texte))

        if fin < len(texte):
            # Essayer de couper à un saut de paragraphe
            coupe = texte.rfind("\n\n", debut, fin)
            if coupe == -1 or coupe <= debut:
                # Sinon couper à la fin d'une phrase
                coupe = max(
                    texte.rfind(". ", debut, fin),
                    texte.rfind("! ", debut, fin),
                    texte.rfind("? ", debut, fin),
                    texte.rfind("\n", debut, fin),
                )
            if coupe > debut:
                fin = coupe + 1

        chunk = texte[debut:fin].strip()
        if len(chunk) >= MIN_CHUNK:
            segments.append({
                "id": f"{source}_{idx}",
                "source": source,
                "texte": chunk,
            })
            idx += 1

        # Chevauchement — debut DOIT toujours avancer
        prochain = fin - CHUNK_OVER if fin < len(texte) else fin
        debut = prochain if prochain > debut else fin

    return segments


# ─── TF-IDF ──────────────────────────────────────────────────────────────────
def calculer_idf(corpus_tokens: list) -> dict:
    """IDF avec lissage de Laplace : log((N+1)/(df+1)) + 1"""
    N = len(corpus_tokens)
    df = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {terme: math.log((N + 1) / (freq + 1)) + 1 for terme, freq in df.items()}


def vecteur_tfidf(tokens: list, idf: dict) -> dict:
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (freq / total) * idf.get(t, 1.0) for t, freq in tf.items()}


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"❌ Dossier '{DATA_DIR}' introuvable. Créez-le et ajoutez vos fichiers .txt.")
        return

    fichiers = list(data_path.glob("*.txt"))
    if not fichiers:
        print(f"❌ Aucun fichier .txt dans '{DATA_DIR}'.")
        return

    print(f"📂 {len(fichiers)} fichier(s) trouvé(s) dans {DATA_DIR}/")

    # 1. Lire et découper
    tous_segments = []
    for f in sorted(fichiers):
        try:
            texte = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            texte = f.read_text(encoding="latin-1")

        segments = decouper_texte(texte, f.stem)
        tous_segments.extend(segments)
        print(f"  ✓ {f.name} → {len(segments)} segments")

    print(f"\n📊 Total : {len(tous_segments)} segments")

    # 2. Sauvegarder la base
    with open(KB_FILE, "w", encoding="utf-8") as fp:
        json.dump(tous_segments, fp, ensure_ascii=False, indent=2)
    print(f"✅ Base sauvegardée : {KB_FILE}")

    # 3. Calculer l'index TF-IDF
    print("⚙️  Calcul de l'index TF-IDF...")
    corpus_tokens = [tokeniser(s["texte"]) for s in tous_segments]
    idf = calculer_idf(corpus_tokens)

    vecteurs = []
    for tokens in corpus_tokens:
        vecteurs.append(vecteur_tfidf(tokens, idf))

    index = {"idf": idf, "vecteurs": vecteurs}
    with open(TFIDF_FILE, "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False)
    print(f"✅ Index TF-IDF sauvegardé : {TFIDF_FILE}")
    print("\n🚀 Prêt ! Lancez maintenant : streamlit run app.py")


if __name__ == "__main__":
    main()
