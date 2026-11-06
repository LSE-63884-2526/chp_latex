"""
signifier_analysis.py
=====================
CHP söylem tezi: tek bir signifier'ın BERT bağlam vektörleri üzerinde
kümeleme, lift ve contextual drift analizi.

Girdi (her terim icin, EMB_DIR altinda):
    {term}tvectors.npy    (N x 768, float32)
    {term}tmetadata.csv   (doc_id, date, year, month, venue, phase, surface, context)

Class terimi BIR KEZ yukler. Butun method'lar self.v ve self.m uzerinde calisir.
Kumeleme tek run'da yapilir; UMAP gorseli ve lift tablosu ayni run'dan uretilir.

Bu dosyaya GIRMEYEN adimlar (ayri concern):
    - BERT embedding uretimi (torch/transformers, ham korpus)  -> ayri script
    - preprocessing                                            -> PreprocessingPipeline
    - KWIC (ham metin uzerinde)                                -> PreprocessingPipeline
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Output Adjustments ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,   # font'u vektor gomer, LaTeX'te temiz cikar
    "ps.fonttype": 42,
})

# ============================================================
# CONFIG  (kendi yollarina gore degistir)
# ============================================================
EMB_DIR       = r"C:\Users\user\Desktop\Thesis\Embeddings"
PLOT_DIR      = r"C:\Users\user\Desktop\Thesis\Plots"
STOP_PATH     = r"C:\Users\user\Desktop\Thesis\stopwords_tr.txt"
ZEMBEREK_JAR  = r"C:\Users\user\Desktop\Thesis\zemberek\zemberek-full.jar"
LATEX_FIG_DIR = r"C:\Users\user\Desktop\Thesis\latex\figures"   # heatmap'ler buraya da mirrorlanir

# ============================================================
# STEM HARITASI (gosterim/ozetleme icin kok birlestirme)
# DEFAULT_STEMS: {regex: stem}. Bir kelime fullmatch ile eslesirse o stem'e
# iner (hakkina/haklarini/hakkimiz -> hak). Eslesmezse root_len ilk-N harf
# fallback (eski davranis). Fit'i, kumelemeyi ETKILEMEZ; sadece kelime
# ozetlerini (summarize, meta_ctfidf, print_group, group_members) birlestirir.
# ============================================================
try:
    from default_stems import DEFAULT_STEMS
    _STEM_COMPILED = [(re.compile(rx), st) for rx, st in DEFAULT_STEMS.items()]
except Exception:
    _STEM_COMPILED = []   # default_stems yoksa: saf root_len davranisi

def to_stem(word, root_len=5, use_stems=True):
    """Kelimeyi koke indir. use_stems=True ve DEFAULT_STEMS'te fullmatch varsa
    tanimli stem'e, yoksa word[:root_len] fallback. use_stems=False ise her
    zaman word[:root_len] (eski davranis)."""
    if use_stems and _STEM_COMPILED:
        for rx, st in _STEM_COMPILED:
            if rx.fullmatch(word):
                return st
    return word[:root_len]


# faz sinirlari (metadata'da phase kolonu yoksa date'ten turetilir)
PHASE_BREAKS = [("2019-04-01", 1), ("2023-11-05", 2)]   # oncesi 1, arasi 2, sonrasi 3

PHASE_COLOR = {"f1": "#228833", "f2": "#CCBB44", "f3": "#4477AA"}  # KK erken / KK gec / Ozel
PHASE_LABEL = {"f1": "Phase 1 (KK Early)", "f2": "Phase 2 (KK Late)", "f3": "Phase 3 (Özel)"}
NEUTRAL     = "#cccccc"

CORPUS_PATH   = r"C:\Users\user\Desktop\Thesis\Data\corpus_main_processed.csv"   
CORPUS_TEXT_COL = "text_bert"        
CORPUS_ID_COL   = None 


# ============================================================
# MODUL SEVIYESI YARDIMCILAR
# ============================================================
def clean_context(text):
    """Apostrof ekini at, Turkce lower."""
    text = re.sub(r"['\u2018\u2019\u02bc]\w+", "", text)
    return text.replace("\u0130", "i").replace("I", "\u0131").lower()


def load_stopwords(path=STOP_PATH):
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip()]


def phase_of(ts):
    """Timestamp -> faz (int). date'ten faz turetmek gerektiginde."""
    if pd.isna(ts):
        return 0
    for cut, ph in PHASE_BREAKS:
        if ts < pd.Timestamp(cut):
            return ph
    return PHASE_BREAKS[-1][1] + 1


# ---- zemberek POS: lazy singleton (JVM bir kez baslar) ----
_MORPH = None


def _get_morphology(jar=ZEMBEREK_JAR):
    global _MORPH
    if _MORPH is None:
        import jpype
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                           "-XX:+SuppressFatalErrorMessage", 
                           f"-Djava.class.path={jar}", convertStrings=False)
        TurkishMorphology = jpype.JClass("zemberek.morphology.TurkishMorphology")
        _MORPH = TurkishMorphology.createWithDefaults()
    return _MORPH


PROPER_KEEP = {"erdoğan", "imamoğlu", "kılıçdaroğlu", "özel", "bahçeli", "kurtuluş", "direniş", "kalkınma",
               "kuruluş", "yürüyüş"}
def is_content_word(word):
    if word.lower() in PROPER_KEEP:
        return True
    try:
        results = _get_morphology().analyze(word)
        analyses = results.getAnalysisResults()
        if analyses.isEmpty():
            return True
        pos = str(analyses.get(0).getDictionaryItem().primaryPos.shortForm)
        return pos in ("Noun", "Adj")
    except Exception:
        return True


def is_verb(word):
    """Fiil (Verb) ise True. Fiiller ayri satirda gosterilecek."""
    try:
        results = _get_morphology().analyze(word)
        analyses = results.getAnalysisResults()
        if analyses.isEmpty():
            return False
        pos = str(analyses.get(0).getDictionaryItem().primaryPos.shortForm)
        return pos == "Verb"
    except Exception:
        return False


# ============================================================
# ANA CLASS
# ============================================================
class SignifierAnalyzer:
    """Tek bir signifier'in BERT bagam vektorleri uzerinde analiz.

    Kullanim:
        a = SignifierAnalyzer("millet", venue=None)
        a.fit(min_cluster_size=100, min_samples=15)     # kumeleme + lift
        a.plot_phase_heatmap()                           # faz-lift heatmap
        a.latex_group_table()                            # meta-cluster tablosu
    """

    def __init__(self, term, venue=None, version="normal", emb_dir=EMB_DIR, stop_path=STOP_PATH,
                 use_stems=True):
        self.term = term
        self.emb_dir = emb_dir
        self.stop = load_stopwords(stop_path)
        self.use_stems = use_stems   # DEFAULT_STEMS ile kok birlestirme (gosterim/ozet)

        valid = {"normal", "maskwhite", "masket", "ctx", "ctx_nostyle"}
        if version not in valid:
            raise ValueError(f"version {version!r} gecersiz; {valid}")
        self.version = version

        # --- load vectors and metadata according to the version ---
        tag = "" if version == "normal" else f"_{version}"
        self.v = np.load(f"{emb_dir}/{term}_cw15vectors{tag}.npy")
        self.m = pd.read_csv(f"{emb_dir}/{term}_cw15metadata{tag}.csv")

        if venue is not None:
            mask = self.m["venue"].str.contains(venue, case=False, na=False).values
            self.v, self.m = self.v[mask], self.m[mask].reset_index(drop=True)
        self.venue = venue

        # exclude duplications
        keep = ~self.m.duplicated(subset=["doc_id", "context"], keep="first").values
        self.v, self.m = self.v[keep], self.m[keep].reset_index(drop=True)

        # compute whitening on context vectors (not target-word)
        if version == "ctx" or version == "ctx_nostyle":
            mu = self.v.mean(0, keepdims=True)
            Xc = self.v - mu
            vals, vecs = np.linalg.eigh(np.cov(Xc, rowvar=False))
            k = min(256, len(self.v) // 5)            
            idx = vals.argsort()[::-1][:k]
            self.v = (Xc @ (vecs[:, idx] / np.sqrt(vals[idx] + 1e-6))).astype(np.float32)

        # date + phase garanti
        if "date" in self.m.columns:
            self.m["date"] = pd.to_datetime(self.m["date"], dayfirst=True, errors="coerce")
        if "phase" not in self.m.columns or self.m["phase"].isna().all():
            self.m["phase"] = self.m["date"].apply(phase_of)

        # kumeleme durumu
        self.clu = None          # 10D kumeleme uzayi
        self._viz = None         # 2D gorsel uzayi (lazy)
        self.out = None          # lift tablosu
        self.cluster_groups = {}  # {"grup adi": [id, id, ...]}  meta-cluster

        self._load_corpus()       # KWIC icin ham korpusu yukle (doc_id -> tam metin)

    def _load_corpus(self):
        """Ham korpusu bir kez yukler, doc_id -> tam metin sozlugu kurar (KWIC icin).

        Korpus okunamazsa (yol yanlis, dosya yok, id eslesmesi bozuk) sessizce
        atlanir; KWIC bu durumda metadata'daki kesik context'e duser (yine calisir).
        """
        if getattr(SignifierAnalyzer, "_corpus_cache", None) is None:
            try:
                raw = pd.read_csv(CORPUS_PATH)
                if CORPUS_ID_COL is None:
                    raw = raw.reset_index(drop=True)          # 0-tabanli index garanti
                    lookup = raw[CORPUS_TEXT_COL].to_dict()    # key = 0,1,2,... = doc_id
                else:
                    lookup = dict(zip(raw[CORPUS_ID_COL], raw[CORPUS_TEXT_COL]))
                SignifierAnalyzer._corpus_cache = lookup   # class-level cache
            except Exception as e:
                print(f"UYARI: korpus yuklenemedi ({e}); KWIC kesik context'e dusecek.")
                SignifierAnalyzer._corpus_cache = {}
        self.corpus_lookup = SignifierAnalyzer._corpus_cache or None
        if not self.corpus_lookup:
            # bos sozluk -> KWIC has_corpus'u False gorsun diye attribute'u kaldir
            if hasattr(self, "corpus_lookup"):
                del self.corpus_lookup
        

    # ----------------------------------------------------------------
    # SENSITIVITY -> CLUSTERING -> LIFT -> WORDS   
    # ----------------------------------------------------------------
    def n_component_sensitivity(self, values=(5, 8, 10, 12, 15, 20),
                            min_cluster_size=None, min_samples=None,
                            venue=None, verbose=False, version=None):
        """Helps to choose n_component parameter for the fit function."""
        term = self.term
        version = self.version if version is None else version
        venue = self.venue if venue is None else venue
        if min_cluster_size is None:
            min_cluster_size = 50 if term == "türkiye" else 15
        if min_samples is None:
            min_samples = max(1, min_cluster_size // 3)
        rows = []
        for nc in values:
            a = SignifierAnalyzer(term, venue=venue, version=version)   # version eklendi
            a.fit(n_components=nc, min_cluster_size=min_cluster_size,
                min_samples=min_samples, verbose=verbose)
            labels = a.m["cluster"]
            n_clust = labels.nunique() - (1 if -1 in labels.values else 0)
            n_noise = (labels == -1).sum()
            noise_pct = round(100 * n_noise / len(labels), 1)
            rows.append({"n_components": nc, "n_clusters": n_clust, "noise_%": noise_pct})
            print(f"  nc={nc:3}  clusters={n_clust:4}  noise={noise_pct}%")
        return pd.DataFrame(rows)

    def fit(self, min_cluster_size=None, min_samples=None, max_df_ratio=1,
            min_word_len=3, n_neighbors=40, top_words=100, min_cluster_freq=5, pos_filter=True,
            ranking="both", root_len=5, verbose=True, dim_ratio=0.85, hide_common=False, n_components=10, random_state=42):
        """..."""
        if min_cluster_size is None:
            min_cluster_size = 50 if self.term == "türkiye" else 15
        if min_samples is None:
            min_samples = max(1, min_cluster_size // 3) # set minimum value to 5
        if verbose:
            print(f"{self.term} | venue={self.venue or 'hepsi'} | n={len(self.v)} | "
                  f"faz: {dict(self.m['phase'].value_counts().sort_index())}")
        if len(self.v) < min_cluster_size * 2:
            print("UYARI: veri az, kumeleme guvenilmez olabilir")

        # --- 10D kumeleme uzayi ---
        self.clu = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=n_components,
                             metric="cosine", random_state=random_state).fit_transform(self.v)
        self.m["cluster"] = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                            min_samples=min_samples).fit_predict(self.clu)
        n_clust = self.m.cluster.nunique() - (1 if -1 in self.m.cluster.values else 0)
        n_noise = (self.m.cluster == -1).sum()
        if verbose:
            print(f"number of clusters: {n_clust} | noise: {100 * n_noise // len(self.m)}%")

        return self._summarize(
            max_df_ratio=max_df_ratio, min_word_len=min_word_len, top_words=top_words,
            min_cluster_freq=min_cluster_freq, pos_filter=pos_filter, ranking=ranking,
            root_len=root_len, verbose=verbose, dim_ratio=dim_ratio, hide_common=hide_common)

    def _summarize(self, max_df_ratio=0.7, min_word_len=3, top_words=100,
                   min_cluster_freq=5, pos_filter=True, ranking="both",
                   root_len=5, verbose=True, dim_ratio=0.85, hide_common=False):
        """fit'in ctfidf + verbs + lift + self.out ureten kismi. Ayri cagrilabilir
        olmasi icin bolundu: split_cluster alt-kume yazdiktan sonra bunu cagirir,
        boylece alt-kumeler (500x) normal kumelerle ayni tabloda, ayni evrende,
        ayni frekans filtreleriyle ozetlenir. self.m['cluster'] neyse ona bakar."""
        # --- kume kelimeleri (ctfidf) ---
        docs = (self.m[self.m.cluster != -1].groupby("cluster")["context"]
                .apply(lambda s: clean_context(" ".join(s))))
        cv = CountVectorizer(stop_words=list(self.stop) + [self.term],
                             token_pattern=rf"(?u)\b[a-z\u00e7\u011f\u0131\u00f6\u015f\u00fc]{{{min_word_len},}}\b",
                             lowercase=False)
        X = cv.fit_transform(docs)
        words = np.array(cv.get_feature_names_out())
        verb_words, verb_counts = None, None
        if pos_filter:
            content_mask = np.array([is_content_word(w) for w in words])
            verb_mask = np.array([is_verb(w) for w in words])
            verb_words = words[verb_mask]
            verb_counts = X[:, verb_mask].toarray()   # kume x fiil (docs = kume metinleri)
            words, X = words[content_mask], X[:, content_mask]

        raw_counts = X.toarray()
        tf = raw_counts.astype(float)
        tf = tf / tf.sum(1, keepdims=True)
        idf = np.log((1 + tf.shape[0]) / (1 + (raw_counts > 0).sum(0)))
        ctfidf = tf * idf
        df_count = (raw_counts > 0).sum(0)
        keep = df_count <= max_df_ratio * tf.shape[0]
        words, ctfidf, raw_counts = words[keep], ctfidf[:, keep], raw_counts[:, keep]

        # soluklastirma icin: hangi kelimeler cok yaygin (kume bazli)
        # kok bazli yayginlik: her kok kac kumede geciyor (cekimleri birlestirerek)
        root_cluster_presence = defaultdict(set)
        for ci in range(raw_counts.shape[0]):
            for i in np.where(raw_counts[ci] >= min_cluster_freq)[0]:
                root_cluster_presence[to_stem(words[i], root_len, self.use_stems)].add(ci)
        n_clusters = raw_counts.shape[0]
        root_dim_global = {r: len(cs) > dim_ratio * n_clusters
                   for r, cs in root_cluster_presence.items()}

        def top_by(score_matrix, ci, use_freq_filter=True, apply_hide=False):
            scores = score_matrix[ci].copy().astype(float)
            freqs = raw_counts[ci]
            if use_freq_filter:
                scores[freqs < min_cluster_freq] = 0
            root_freq, root_best, root_score = defaultdict(int), {}, defaultdict(float)
            for i in np.where(scores > 0)[0]:
                w, root = words[i], to_stem(words[i], root_len, self.use_stems)
                root_freq[root] += int(freqs[i])
                root_score[root] += scores[i]
                if root not in root_best or freqs[i] > root_best[root][1]:
                    root_best[root] = (w, freqs[i])
            roots = sorted(root_score, key=root_score.get, reverse=True)[:top_words]
            roots = sorted(roots, key=lambda r: root_freq[r], reverse=True)
            out_words = []
            for rank, r in enumerate(roots):
                if root_dim_global.get(r, False) and apply_hide:   # yaygin kok, gizle
                    continue
                prefix = "·" if root_dim_global.get(r, False) else ""   # yayginsa nokta
                if rank < 10:
                    label = f"{prefix}{root_best[r][0]}({root_freq[r]}; {root_best[r][1]})"
                else:
                    label = f"{prefix}{root_best[r][0]}({root_freq[r]})"
                out_words.append(label)
            return out_words
        # --- lift tablosu ---
        phase_share = self.m["phase"].value_counts(normalize=True).to_dict()
        rows = []
        for ci, c in enumerate(docs.index):
            cl = self.m[self.m.cluster == c]
            lifts = {ph: (cl["phase"] == ph).mean() / phase_share.get(ph, 1e-9)
                     for ph in [1, 2, 3]}
            dom = max(lifts, key=lifts.get)
            row = {"cluster": c, "n": len(cl),
                   "lift1": lifts[1], "lift2": lifts[2], "lift3": lifts[3],
                   "dom_phase": dom, "dom_lift": lifts[dom]}
            if ranking in ("freq", "both"):
                row["freq_words"] = ", ".join(top_by(raw_counts, ci, use_freq_filter=False, apply_hide=False))
            if ranking in ("ctfidf", "both"):
                row["ctfidf_words"] = ", ".join(top_by(ctfidf, ci, apply_hide=hide_common))
            # en sik fiiller (ayri satir): kok gruplu, en sik cekim + kok toplam
            if verb_counts is not None:
                vfreq, vbest = defaultdict(int), {}
                for i in np.where(verb_counts[ci] >= min_cluster_freq)[0]:
                    vr = to_stem(verb_words[i], root_len, self.use_stems)
                    vfreq[vr] += int(verb_counts[ci][i])
                    if vr not in vbest or verb_counts[ci][i] > vbest[vr][1]:
                        vbest[vr] = (verb_words[i], int(verb_counts[ci][i]))
                vtop = sorted(vfreq, key=vfreq.get, reverse=True)[:20]
                row["verb_words"] = ", ".join(f"{vbest[r][0]}({vfreq[r]})" for r in vtop)
            else:
                row["verb_words"] = ""
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("dom_lift", ascending=False).reset_index(drop=True)

        if verbose:
            for _, r in out.iterrows():
                head = (f"c{r['cluster']:>2} n={r['n']:>4} ")
                if ranking == "both":
                    print(f"{head}\n   freq: {r['freq_words']}\n   ctfidf: {r['ctfidf_words']}"
                          f"\n   verbs: {r['verb_words']}")
                elif ranking == "freq":
                    print(f"{head} | {r['freq_words']}\n   verbs: {r['verb_words']}")
                else:
                    print(f"{head} | {r['ctfidf_words']}\n   verbs: {r['verb_words']}")

        out["dom_phase"] = out["dom_phase"].apply(lambda x: f"f{x}")
        self.out = out
        # meta_words icin: kelime bazli freq matrisi + satir->kume eslemesi
        self._meta_raw = raw_counts          # kume x kelime ham sayim
        self._meta_words = words             # kelime dizisi (raw_counts sutunlariyla ayni)
        self._meta_rowmap = {int(c): ci for ci, c in enumerate(docs.index)}  # kume id -> satir
        return self

    def _ensure_fit(self):
        if self.out is None or "cluster" not in self.m.columns:
            raise RuntimeError("once .fit() cagir")


    # ----------------------------------------------------------------
    # UMAP GORSEL   (cell 5 + 6)
    # ----------------------------------------------------------------

    def set_cluster_names(self, mapping):
        """cluster_id -> nitel kod adi eslestir. grup adlari (n=...) ile birlikte kullanilir.
        Ust uste cagrilabilir, birikir. Ornek:
            a.set_cluster_names({12: "Kemalist Populism", 4: "Halk Egemenligi"})
        """
        self.cluster_names = {**getattr(self, "cluster_names", {}), **mapping}
        return self

    def set_cluster_group(self, name, cluster_ids):
        """Birden cok cluster'i tek meta-cluster olarak grupla.
        Gorselde tek renk + ortalanmis tek etiket, tabloda birlesik n/lift.
        Tek elemanli liste de olur. Ust uste cagrilabilir, birikir.
            a.set_cluster_group("Populist Mobilisation", [19,29,30,31,32,33,34,35,36])
        """
        self.cluster_groups[name] = [int(c) for c in cluster_ids]
        return self

    def meta_lift(self, cluster_ids, verbose=True):
        """Bir kume grubunun birlesik lift'i (alt-kumeleri tek kume sayarak).
        Lift'leri ORTALAMAZ; uyeleri toplayip bastan hesaplar.
            a.meta_lift([19,29,30,31,32,33,34,35,36])
        Doner: {"n", "lift1", "lift2", "lift3"}
        """
        self._ensure_fit()
        ids = [int(c) for c in cluster_ids]
        mask = self.m["cluster"].isin(ids)
        total = int(mask.sum())
        phase_share = self.m["phase"].value_counts(normalize=True).to_dict()
        meta_ph = self.m.loc[mask, "phase"]
        lifts = {ph: (meta_ph == ph).mean() / phase_share.get(ph, 1e-9) for ph in [1, 2, 3]}
        if verbose:
            print(f"meta n={total} | f1={lifts[1]:.1f} f2={lifts[2]:.1f} f3={lifts[3]:.1f}")
        return {"n": total, "lift1": lifts[1], "lift2": lifts[2], "lift3": lifts[3]}
    
    def all_meta_lifts(self, verbose=False):
        """Tüm meta-kümelerin birleşik n + faz lift'lerini DataFrame döndürür."""
        rows = []
        for name, ids in self.cluster_groups.items():
            r = self.meta_lift(ids, verbose=verbose)
            rows.append({"group": name, **r})
        return pd.DataFrame(rows).set_index("group")

    def plot_phase_heatmap(self, save_dir=None, show=True, fmt="pdf",
                            dpi=300, figsize=(9, 7), cmap="RdBu_r",
                            sort_by_centroid=True, label_fs=12,
                            flat_thresh=1.35, log_lim=2.0, text_thresh=1.1,
                            min_n=0, count="docs",
                            phase_labels=("KK-Erken\n(2015-2019)",
                                            "KK-Geç\n(2019-2023)",
                                            "Özel\n(2023-2026)")):
            """Meta-küme faz-lift ısı haritası. all_meta_lifts() çıktısını kullanır.

            Renk log2(lift) üzerinden: 0.5 ve 2.0 simetrik görünür, 1.0 beyaz.
            Hücrelerde yine ham lift yazar.
            log_lim: renk ölçeğinin sınırı (log2). 2.0 -> 0.25 ile 4 arası.
            text_thresh: |log2(lift)| bu eşiği aşınca yazı beyaz, aşmayınca siyah.
            min_n: bu sayının altındaki gruplar çizilmez (0 = hepsi çizilir).
            count: satır etiketinde ne yazsın. 'docs' = farklı konuşma sayısı,
                'n' = gözlem sayısı, None = hiçbir şey.
                a.plot_phase_heatmap(save_dir="figures", show=False, min_n=20)
            """
            import seaborn as sns
            df = self.all_meta_lifts(verbose=False)

            # --- grup başına konuşma sayısı (örtüşen pencereleri tek sayar) ---
            docs = {name: self.m.loc[self.m["cluster"].isin([int(c) for c in ids]),
                                    "doc_id"].nunique()
                    for name, ids in self.cluster_groups.items()}
            df["docs"] = pd.Series(docs)

            # --- küçük grupları ele ---
            size_col = "docs" if count == "docs" else "n"
            if min_n:
                dropped = df.index[df[size_col] < min_n].tolist()
                df = df[df[size_col] >= min_n]
                if dropped:
                    print(f"min_n={min_n} altında kalan, çizilmeyen: {dropped}")

            m = df[["lift1", "lift2", "lift3"]].copy()
            m.columns = list(phase_labels)

            # --- sıralama (ham lift üzerinden) ---
            if sort_by_centroid:
                vals = m.values
                dom = vals.argmax(axis=1)                        # 0=Erken, 1=Geç, 2=Özel
                peak = vals.max(axis=1)
                group = np.where(peak < flat_thresh, 1.5, dom)   # düz satırlar Geç ile Özel arasına
                tot = vals.sum(axis=1)
                centroid = np.where(tot > 0, vals @ np.array([1, 2, 3]) / tot, 2.0)
                # öncelik: grup, sonra merkez (artan), eşitlikte tepe (azalan)
                order = np.lexsort((-peak, centroid, group))
                m = m.iloc[order]
                df = df.loc[m.index]

            # --- satır etiketleri: tırnak temizliği + n ---
            labels = []
            for name in m.index:
                clean = str(name).strip("'\"")
                if count == "docs":
                    labels.append(f"{clean} ({df.at[name, 'docs']})")
                elif count == "n":
                    labels.append(f"{clean} (n={df.at[name, 'n']})")
                else:
                    labels.append(clean)

            # --- renk log2 üzerinden ---
            floor = 2 ** -log_lim
            color_m = np.log2(m.clip(lower=floor))

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(color_m, annot=False, cmap=cmap, center=0,
                        vmin=-log_lim, vmax=log_lim,
                        linewidths=0.5, linecolor="white",
                        yticklabels=labels, ax=ax)

            # --- sayılar: ham lift, yazı rengi log2 uzaklığına göre ---
            for yi in range(m.shape[0]):
                for xi in range(m.shape[1]):
                    val = m.values[yi, xi]
                    dist = abs(color_m.values[yi, xi])
                    color = "white" if dist >= text_thresh else "black"
                    ax.text(xi + 0.5, yi + 0.5, f"{val:.1f}",
                            ha="center", va="center",
                            color=color, fontsize=label_fs)

            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.tick_params(labelsize=label_fs)
            ax.set_ylabel(""); ax.set_xlabel("")

            # --- renk çubuğu: log2 konumlarında ham lift değerleri ---
            cbar = ax.collections[0].colorbar
            ticks = np.arange(-int(log_lim), int(log_lim) + 1)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{2.0 ** t:g}" for t in ticks])
            cbar.set_label("Göreli yoğunluk (1 = ortalama)", size=label_fs - 1)
            cbar.ax.tick_params(labelsize=label_fs - 2)

            plt.tight_layout()

            if save_dir:
                targets = [save_dir]
                if LATEX_FIG_DIR and os.path.abspath(LATEX_FIG_DIR) != os.path.abspath(save_dir):
                    targets.append(LATEX_FIG_DIR)
                for d in targets:
                    os.makedirs(d, exist_ok=True)
                    path = f"{d}/{self.term}_phase_heatmap.{fmt}"
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    print("saved:", path)
            if show:
                plt.show()
            else:
                plt.close(fig)
            return self
    
    def meta_words(self, cluster_ids, top=30, root_len=None, verbose=True):
        """Bir kume grubunun BIRLESIK freq listesi (alt-kumeleri toplayarak).
        Her kelime kok bazinda gruplanir, alt-kumelerdeki freq'ler TOPLANIR,
        birlesik frekans sirali liste doner. Tek kume freq gosterimiyle ayni
        mantik (en sik cekim + kok toplam), sadece grup uzerinden.
            a.meta_words([19,29,30,31,32,33,34,35,36], top=25)
        Doner: DataFrame [form, freq]  ve verbose ise tek satir yazdirir.
        """
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("meta_words icin fit'i yeniden calistir (eski fit ciktisinda "
                               "meta verisi saklanmamis)")
        rl = root_len if root_len is not None else getattr(self, "_root_len", 5)
        ids = [int(c) for c in cluster_ids]
        rows_idx = [self._meta_rowmap[c] for c in ids if c in self._meta_rowmap]
        if not rows_idx:
            print("bu id'ler icin satir bulunamadi:", ids)
            return pd.DataFrame(columns=["form", "freq"])

        # grup ici toplam: her kelime icin secili kumelerdeki freq toplami
        summed = self._meta_raw[rows_idx].sum(axis=0)   # kelime basina birlesik freq

        # kok bazinda grupla: kok toplam + o kokun en sik cekimi
        root_freq, root_best = defaultdict(int), {}
        for i in np.where(summed > 0)[0]:
            w = self._meta_words[i]
            root = to_stem(w, rl, self.use_stems)
            f = int(summed[i])
            root_freq[root] += f
            if root not in root_best or f > root_best[root][1]:
                root_best[root] = (w, f)

        roots = sorted(root_freq, key=root_freq.get, reverse=True)[:top]
        out_rows = [(root_best[r][0], root_freq[r]) for r in roots]
        df = pd.DataFrame(out_rows, columns=["form", "freq"])
        if verbose:
            print(f"meta ({', '.join('c'+str(c) for c in ids)}) birlesik freq:")
            print("   " + ", ".join(f"{f}({n})" for f, n in out_rows))
        return df

    def meta_ctfidf(self, top=30, groups=None, verbose=True, reduce_frequent=False,
                    shared_min_share=0.8, shared_penalty=0.5, keep=None):
        """Her meta-kumeyi DIGER meta-kumelere karsi ayirt eden kelimeler.
        corpus = meta-kumeler; ct-tfidf bu 10 'dokuman' uzerinden yeniden hesaplanir.
        reduce_frequent:
          False  -> hicbir bastirma (ham ct-tfidf).
          True   -> tf'nin karekokunu HERKESE uygular (tum kelimeler bastirilir).
          'shared' -> SADECE shared koklerin ct-tfidf skorunu shared_penalty ile
                      carpar (dusurur). Yani sadece kumelerin cogunda gecen yaygin
                      kokler (parti, cumhuriyet, devlet) bastirilir; kumeye-ozgu
                      kokler (lira, ciftci, kul) hic dokunulmadan kalir.
                      shared_min_share: bir kok kac kumede gecerse 'shared' sayilir.
                      shared_penalty: shared koke uygulanan skor carpani (0.5 = yariya).
        keep: shared olsa bile CEZALANMAYACAK kelimeler (orn: ['erdoğan']).
              Analitik olarak merkezi ama yaygin ozel isimleri korumak icin.
        Doner: {grup: DataFrame[form, ctfidf, freq]}"""
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("meta_ctfidf icin fit'i yeniden calistir")
        groups = groups or self.cluster_groups
        names = list(groups)
        rl = getattr(self, "_root_len", 5)

        # 1) her meta-kume icin birlesik ham freq vektoru (kelime bazli)
        M = np.zeros((len(names), self._meta_raw.shape[1]))
        for gi, name in enumerate(names):
            rows = [self._meta_rowmap[int(c)] for c in groups[name]
                    if int(c) in self._meta_rowmap]
            M[gi] = self._meta_raw[rows].sum(axis=0)

        # 2) ct-tfidf: tf (satir-ici oran) * idf (kac meta-kumede geciyor)
        #    idf +1'li: yaygin kokler (turkiye, lira) sifira INMEZ. Bir kok
        #    baskin oldugu kumede yuksek tf ile kalir, marjinal kumelerde
        #    ct-tfidf dusuk olur. Cok yaygin gostergeler (turkiye, cumhuriyet)
        #    ayrica shared_roots ile 'shared across clusters' satirinda toplanir.
        tf = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)
        if reduce_frequent is True:
            tf = np.sqrt(tf)          # herkese karekok (eski davranis)
        df_count = (M > 0).sum(axis=0)
        idf = np.log((len(names) + 1) / (df_count + 1)) + 1
        ctfidf = tf * idf

        if reduce_frequent == "shared":
            # kok bazli: her kok kac kumede geciyor (cekimleri birlestirerek)
            root_cluster = defaultdict(set)
            for gi in range(M.shape[0]):
                for i in np.where(M[gi] > 0)[0]:
                    root_cluster[to_stem(self._meta_words[i], rl, self.use_stems)].add(gi)
            thresh = shared_min_share * len(names)
            shared_roots_set = {r for r, cs in root_cluster.items()
                                if len(cs) >= thresh}
            # keep'teki kelimeleri kok bazina indirip shared'dan cikar (cezalanmasin)
            keep_roots = {to_stem(w, rl, self.use_stems) for w in (keep or [])}
            shared_roots_set -= keep_roots
            # SADECE shared koklerin skorunu cezala (dusur), digerlerine dokunma
            shared_cols = np.array([to_stem(self._meta_words[i], rl, self.use_stems) in shared_roots_set
                                    for i in range(M.shape[1])])
            ctfidf[:, shared_cols] *= shared_penalty

        # 3) her grup icin kok bazinda en yuksek ct-tfidf
        result = {}
        for gi, name in enumerate(names):
            root_score, root_best, root_freq = defaultdict(float), {}, defaultdict(int)
            for i in np.where(M[gi] > 0)[0]:
                w = self._meta_words[i]; root = to_stem(w, rl, self.use_stems)
                root_score[root] += ctfidf[gi, i]
                root_freq[root] += int(M[gi, i])
                if root not in root_best or M[gi, i] > root_best[root][1]:
                    root_best[root] = (w, int(M[gi, i]))
            roots = sorted(root_score, key=root_score.get, reverse=True)[:top]
            result[name] = pd.DataFrame(
                [(root_best[r][0], round(root_score[r], 3), root_freq[r]) for r in roots],
                columns=["form", "ctfidf", "freq"])
            if verbose:
                print(f"{name}: " + ", ".join(f"{f}" for f in result[name]['form'][:12]))
        return result

    def shared_roots(self, groups=None, min_share=0.9, top=10):
        """Meta-kumelerin cogunda gecen kokler (ct-tfidf ile bastirilan yaygin
        gostergeler: turkiye, cumhuriyet, millet...). Tablo altinda 'shared across
        clusters' satiri icin. min_share: bir kok kac grubun oraninda gecerse
        'shared' sayilir. Doner: DataFrame[form, freq, n_groups].
            a.shared_roots(min_share=0.8)"""
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("shared_roots icin fit'i yeniden calistir")
        groups = groups or self.cluster_groups
        names = list(groups)
        rl = getattr(self, "_root_len", 5)

        # her grup icin birlesik freq (meta_ctfidf ile ayni M)
        M = np.zeros((len(names), self._meta_raw.shape[1]))
        for gi, name in enumerate(names):
            rows = [self._meta_rowmap[int(c)] for c in groups[name]
                    if int(c) in self._meta_rowmap]
            M[gi] = self._meta_raw[rows].sum(axis=0)

        # kok bazinda: kac grupta geciyor + toplam freq + en sik cekim
        root_groups, root_freq, root_best = defaultdict(set), defaultdict(int), {}
        for gi in range(len(names)):
            for i in np.where(M[gi] > 0)[0]:
                w = self._meta_words[i]; root = to_stem(w, rl, self.use_stems)
                root_groups[root].add(gi)
                root_freq[root] += int(M[gi, i])
                if root not in root_best or M[gi, i] > root_best[root][1]:
                    root_best[root] = (w, int(M[gi, i]))

        thresh = min_share * len(names)
        shared = [(root_best[r][0], root_freq[r], len(gs))
                  for r, gs in root_groups.items() if len(gs) >= thresh]
        # signifier'in kendisini disla
        shared = [row for row in shared
                  if to_stem(row[0], rl, self.use_stems) != to_stem(self.term, rl, self.use_stems)]
        shared.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(shared[:top], columns=["form", "freq", "n_groups"])

    def group_members(self, name=None, words=None, top=30):
        """name yoksa {grup: [id...]} tüm harita.
        name varsa o grubun id listesi.
        words="freq"    -> (ids, meta_words DataFrame)
        words="ctfidf"  -> (ids, o grubun meta_ctfidf DataFrame'i)
        words="both"    -> (ids, freq_df, ctfidf_df)"""
        if name is None:
            return dict(self.cluster_groups)
        ids = self.cluster_groups[name]
        if words is None:
            return ids
        if words == "freq":
            return ids, self.meta_words(ids, top=top, verbose=False)
        if words == "ctfidf":
            return ids, self.meta_ctfidf(top=top, verbose=False)[name]
        if words == "both":
            return (ids,
                    self.meta_words(ids, top=top, verbose=False),
                    self.meta_ctfidf(top=top, verbose=False)[name])
        raise ValueError("words: None | 'freq' | 'ctfidf' | 'both'")

    def latex_group_table(self, top_words=20, reduce_frequent=False,
                          shared_penalty=0.5, keep=None):
        """Elle-doldurulacak LaTeX iskeleti (Turkce terimlerle).
        Sen ceviri + \\ant{} + \\midrule eklersin. Id'ler appendixte."""
        cti = self.meta_ctfidf(top=top_words, verbose=False, reduce_frequent=reduce_frequent,
                               shared_penalty=shared_penalty, keep=keep)
        order = []
        for name, ids in self.cluster_groups.items():
            r = self.meta_lift(ids, verbose=False)
            lifts = [r["lift1"], r["lift2"], r["lift3"]]
            dom = int(np.argmax(lifts))
            # faz agirlik merkezi: dusuk = KK-Early, yuksek = Ozel
            phase_idx = np.array([1, 2, 3])
            centroid = (np.array(lifts) * phase_idx).sum() / sum(lifts)
            order.append((name, ids, lifts, dom, centroid))
        order.sort(key=lambda x: x[4])   # centroid'e gore: KK-Early ustte, Ozel altta

        header = (
            "\\footnotesize\n"
            "\\begin{longtable}{l r r r p{7cm}}\n"
            "\\caption{Contextual clusters of \\textit{hak}...} \\\\\n"
            "\\toprule\n"
            "\\textbf{Cluster (n)} & \\textbf{KK-E} & \\textbf{KK-L} & "
            "\\textbf{Özel} & \\textbf{Representative terms (freq.)} \\\\\n"
            "\\midrule\n"
            "\\endfirsthead\n\n"
            "\\multicolumn{5}{c}{\\tablename\\ \\thetable\\ \\textit{(devamı)}} \\\\\n"
            "\\toprule\n"
            "\\textbf{Cluster (n)} & \\textbf{KK-E} & \\textbf{KK-L} & "
            "\\textbf{Özel} & \\textbf{Representative terms (freq.)} \\\\\n"
            "\\midrule\n"
            "\\endhead\n\n"
            "\\bottomrule\n"
            "\\endlastfoot\n"
        )

        lines = []
        for name, ids, lifts, dom, _ in order:
            n_total = int(self.out[self.out["cluster"].isin(ids)]["n"].sum())
            lift_cells = [f"\\textbf{{{lv:.1f}}}" if i == dom else f"{lv:.1f}"
                        for i, lv in enumerate(lifts)]
            tbl = cti[name].head(top_words)          # ctfidf skoruna gore SEC ve SIRALA
            terms = []
            for j, (_, row) in enumerate(tbl.iterrows()):
                form, freq = row["form"], int(row["freq"])
                terms.append(f"{form} ({freq})")
            lines.append(
                f"\\raggedright \\textbf{{{name}}} \\newline ({n_total})\n"
                f"  & {lift_cells[0]} & {lift_cells[1]} & {lift_cells[2]}\n"
                f"  & {', '.join(terms)} \\\\[8pt]"
            )

        body = "\n".join(lines)

        # --- shared across clusters satiri ---
        shared = self.shared_roots(min_share=0.8, top=15)
        if not shared.empty:
            shared_terms = ", ".join(f"{r['form']} ({int(r['freq'])})"
                                     for _, r in shared.iterrows())
            body += (f"\n\\midrule\n"
                     f"\\raggedright \\textit{{Shared across clusters}}\n"
                     f"  & & &\n"
                     f"  & {shared_terms} \\\\")

        footer = "\n\\end{longtable}"
        return header + "\n" + body + footer

    def latex_group_table_short(self, picked=None, top_words=6, signifier="hak",
                                reduce_frequent=False, shared_penalty=0.5, keep=None):
        """Short main-text version: only hand-picked clusters, fewer terms.
        'picked' = list of cluster_group names to include, in the order you want.
        If None, uses all groups. Appendix keeps the full longtable."""
        cti = self.meta_ctfidf(top=top_words, verbose=False, reduce_frequent=reduce_frequent,
                               shared_penalty=shared_penalty, keep=keep)

        names = picked if picked is not None else list(self.cluster_groups.keys())

        header = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\footnotesize\n"
            f"\\caption{{Selected contextual clusters of \\sig{{{signifier}}} "
            "and their phase-specific lift. "
            "Terms are selected and ordered by c-TF-IDF score; "
            "parenthetical counts are raw frequencies (inflections merged).}\n"
            f"\\label{{tab:clusters-{signifier}}}\n"
            "\\vspace{0.5em}\n"
            "\\begin{tabular}{p{2.5cm} r r r p{8cm}}\n"
            "\\toprule\n"
            "\\textbf{Cluster (n)} & \\textbf{KK-E} & \\textbf{KK-L} & "
            "\\textbf{Özel} & \\textbf{Representative terms (freq.)} \\\\\n"
            "\\midrule\n"
        )

        lines = []
        for name in names:
            ids = self.cluster_groups[name]
            r = self.meta_lift(ids, verbose=False)
            lifts = [r["lift1"], r["lift2"], r["lift3"]]
            dom = int(np.argmax(lifts))
            n_total = int(self.out[self.out["cluster"].isin(ids)]["n"].sum())
            lift_cells = [f"\\textbf{{{lv:.1f}}}" if i == dom else f"{lv:.1f}"
                        for i, lv in enumerate(lifts)]
            tbl = cti[name].head(top_words)          # ctfidf skoruna gore SEC ve SIRALA
            terms = []
            for j, (_, row) in enumerate(tbl.iterrows()):
                form, freq = row["form"], int(row["freq"])
                terms.append(f"{form} ({freq})")
            lines.append(
                f"\\raggedright \\textbf{{{name}}} ({n_total})\n"
                f"  & {lift_cells[0]} & {lift_cells[1]} & {lift_cells[2]}\n"
                f"  & {', '.join(terms)} \\\\\n\\addlinespace"
            )

        body = "\n".join(lines)

        # --- shared across clusters satiri: ct-tfidf ile bastirilan yaygin gostergeler ---
        shared = self.shared_roots(min_share=0.8, top=15)
        if not shared.empty:
            shared_terms = ", ".join(f"{r['form']} ({int(r['freq'])})"
                                     for _, r in shared.iterrows())
            shared_row = (f"\n\\midrule\n"
                          f"\\raggedright \\textit{{Shared across clusters}}\n"
                          f"  & & &\n"
                          f"  & {shared_terms} \\\\")
        else:
            shared_row = ""

        footer = shared_row + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
        return header + body + footer

    def print_group(self, name, ranking="both"):
        """Bir meta-kumedeki alt-kumeleri TEKER TEKER, fit ciktisiyla ayni
        formatta basar (n, faz lift'leri, freq/ctfidf kelimeleri, fiiller).
            a.print_group("Moral Node")
        """
        self._ensure_fit()
        ids = [int(c) for c in self.cluster_groups[name]]
        sub = self.out[self.out["cluster"].isin(ids)].copy()
        if sub.empty:
            print(f"'{name}' icin kume bulunamadi (fit guncel mi?)")
            return

        # grup ozeti ustte
        print(f"=== {name} | {len(sub)} kume | toplam n={int(sub['n'].sum())} ===")
        for _, r in sub.iterrows():
            dph = str(r["dom_phase"]).lstrip("f")
            head = (f"c{r['cluster']:>2} n={r['n']:>4} "
                    f"f1={r['lift1']:.1f} f2={r['lift2']:.1f} f3={r['lift3']:.1f} "
                    f"[f{dph} {r['dom_lift']:.1f}]")
            if ranking == "both":
                print(f"{head}\n   freq: {r.get('freq_words','')}\n"
                    f"   ctfidf: {r.get('ctfidf_words','')}\n"
                    f"   verbs: {r.get('verb_words','')}")
            elif ranking == "freq":
                print(f"{head} | {r.get('freq_words','')}\n   verbs: {r.get('verb_words','')}")
            else:
                print(f"{head} | {r.get('ctfidf_words','')}\n   verbs: {r.get('verb_words','')}")

    def forms_in_cluster(self, cluster, keyword=None, root_len=5, top=30, verbose=True):
        """Bir kumede (ya da meta-cluster'da) tek bir kokun butun cekimlerini
        frekans sirali gosterir. Tabloya hangi cekimi yazacagina karar vermek icin.

        cluster: tek kume id (int) VEYA kume id listesi (meta-cluster, cekimler
        alt-kumeler boyunca TOPLANIR).
        keyword: kok kelimesi (None ise self.term). root_len ilk N harf = kok grubu.
            forms_in_cluster(5, "demokrasi")            # tek kume
            forms_in_cluster([19,29,30], "demokrat")    # meta-cluster (toplam)
        Doner: DataFrame [form, freq]. NOT: sadece fit'teki min_word_len +
        content-word filtresinden gecen cekimleri gorur (meta_words ile ayni kaynak).
        """
        self._ensure_fit()
        if not hasattr(self, "_meta_raw"):
            raise RuntimeError("forms_in_cluster icin fit'i yeniden calistir "
                               "(eski fit ciktisinda cekim verisi saklanmamis)")
        ids = [int(cluster)] if isinstance(cluster, (int, np.integer)) else [int(c) for c in cluster]
        rows_idx = [self._meta_rowmap[c] for c in ids if c in self._meta_rowmap]
        if not rows_idx:
            print("bu id'ler icin satir bulunamadi:", ids)
            return pd.DataFrame(columns=["form", "freq"])

        key = clean_context(keyword if keyword is not None else self.term)
        grp = key[:root_len]

        # secili kume(ler) boyunca her kelimenin toplam freq'i
        summed = self._meta_raw[rows_idx].sum(axis=0)   # kelime basina birlesik freq
        # sadece bu kokle baslayan cekimleri topla
        rows = []
        for i in np.where(summed > 0)[0]:
            w = self._meta_words[i]
            if w[:root_len] == grp:                      # ayni kok grubu
                rows.append((w, int(summed[i])))
        rows.sort(key=lambda x: x[1], reverse=True)
        rows = rows[:top]

        df = pd.DataFrame(rows, columns=["form", "freq"])
        if verbose:
            tag = f"c{ids[0]}" if len(ids) == 1 else f"meta({','.join('c'+str(c) for c in ids)})"
            if rows:
                print(f"{tag} '{grp}' cekimleri:  " + ",  ".join(f"{f}({n})" for f, n in rows))
            else:
                print(f"{tag}'de '{grp}' koku yok")
        return df

    def forms_across_clusters(self, form="hakkını", exact=True):
        """Bir yuzey formunun her kumede kac kez gectigini gosterir.
        exact=True: tam esleme (hakkını == hakkını)
        exact=False: substring (hakkını, hakkınızı, hakkınıza hepsi sayilir)
        """
        self._ensure_fit()
        f = clean_context(form)
        # surface birlestirilmis occurrence'larda coklu form tutabilir
        # ("hakları | hakları | hakkını"); her mention ayri kontrol edilir.
        def _match(s):
            parts = [p.strip() for p in clean_context(str(s)).split("|")]
            if exact:
                return any(p == f for p in parts)
            return any(f in p for p in parts)
        hit = self.m["surface"].map(_match)

        rows = []
        for c in sorted(x for x in self.m.cluster.unique() if x != -1):
            mask = (self.m.cluster == c)
            n_tot = int(mask.sum())
            n_form = int((mask & hit).sum())
            rows.append({
                "cluster": c,
                "n_total": n_tot,
                f"n_{f}": n_form,
                "ratio": round(n_form / n_tot, 3) if n_tot else 0.0,
            })
        df = pd.DataFrame(rows).sort_values(f"n_{f}", ascending=False).reset_index(drop=True)
        total_form = int(hit.sum())
        print(f"'{f}' toplam {total_form} kez, {(df[f'n_{f}']>0).sum()} kümede geçiyor")
        return df


    # ----------------------------------------------------------------
    # ALT-KUMELEME   (soz dizimsel buyuk kumeyi soylemsel eksenlere bolme)
    # ----------------------------------------------------------------
    @staticmethod
    def fmt_id(cid):
        """Alt-kume id'sini okunabilir gosterir. Sema: base = 1000 + parent*100.
        1503 -> '5;3', 1000 -> '0;0', 6012 -> '50;12'. Normal id (< 1000) oldugu
        gibi kalir. Rapor/tablo icin."""
        cid = int(cid)
        if cid >= 1000:
            b = cid - 1000
            p = "N" if b // 100 == 999 else b // 100
            return f"{p};{b % 100}"
        return str(cid)

    def subcluster(self, cluster_id, n_components=12, min_cluster_size=15,
                   min_samples=5, show=None, n_neighbors=40):
        """Bir kumeyi alt-bol ve OKU. self.m'yi kalici DEGISTIRMEZ; alt-kumeleri
        gecici yazip _summarize ile fit formatinda (ctfidf + freq + lift + verbs,
        TUM corpus'a kiyasla) basar, sonra self.m'yi eski haline dondurur.
        Amac: hangi alt-kumelerin anlamli oldugunu fit ile ayni olcutle secip,
        ayni parametrelerle split_cluster'a keep=[...] vermek."""
        mask = (self.m.cluster == cluster_id).values
        sub_v = self.v[mask]
        idx = self.m.index[mask]
        print(f"c{cluster_id}: {mask.sum()} occurrence alt-kumeleniyor")
        if len(sub_v) < min_cluster_size * 2:
            print("cok kucuk, alt-kumeleme anlamsiz")
            return None

        clu = umap.UMAP(min_dist=0.0, n_components=n_components,
                        metric="cosine", random_state=42, n_neighbors=n_neighbors).fit_transform(sub_v)
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=min_samples).fit_predict(clu)
        n_sub = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).mean()
        print(f"  -> {n_sub} alt-kume, noise {noise:.0%}\n")

        # --- gecici yaz: alt-kumeler 1000+ id ile, tutulmayan degil hepsi ---
        base = 1000 + (999 if cluster_id == -1 else cluster_id) * 100
        saved_out = self.out                      # mevcut tabloyu sakla
        saved_labels = self.m.loc[idx, "cluster"].copy()   # orijinal etiketler
        for orig_i, lab in zip(idx, labels):
            self.m.at[orig_i, "cluster"] = (base + lab) if lab != -1 else -1

        # --- fit formatinda ozetle ve SADECE alt-kumeleri bas ---
        self._summarize(ranking="both", verbose=False)
        sub_ids = sorted(c for c in self.out.cluster.unique() if c >= base and c < base + 100)
        # buyukten kucuge
        sub_rows = self.out[self.out.cluster.isin(sub_ids)].sort_values("n", ascending=False)
        order = sub_rows if show is None else sub_rows.head(show)
        for _, r in order.iterrows():
            raw = int(r["cluster"]) - base       # keep=[...] icin ham label
            faz = self.m[self.m.cluster == r["cluster"]]["phase"].value_counts().sort_index().to_dict()
            print(f"  alt-kume {raw} (n={r['n']}, faz={faz}) "
                  f"f1={r['lift1']:.1f} f2={r['lift2']:.1f} f3={r['lift3']:.1f} "
                  f"[{r['dom_phase']} {r['dom_lift']:.1f}]")
            if r.get("freq_words"):
                print(f"     freq: {r['freq_words']}")
            if r.get("ctfidf_words"):
                print(f"     ctfidf: {r['ctfidf_words']}")
            print(f"     verbs: {r['verb_words']}\n")

        # --- geri al: self.m ve self.out eski haline ---
        self.m.loc[idx, "cluster"] = saved_labels
        self.out = saved_out
        return None

    def split_cluster(self, cluster_id, keep, n_components=12,
                      min_cluster_size=15, min_samples=5, resummarize=True, n_neighbors=40):
        """Kumeyi alt-bol ve SADECE keep'teki alt-kumeleri ana cluster kolonuna
        yeni id (cluster_id*100 + sub) olarak yaz. Tutulmayan her sey noise'a (-1)
        gider. Sonra _summarize ile tum tablolari (ctfidf/verbs/lift/self.out)
        yeniler; boylece alt-kumeler normal kumelerle ayni tabloda, ayni evrende
        gorunur ve meta-cluster'a normal kume gibi verilebilir.

        keep: tutulacak alt-kume numaralari. subcluster ile AYNI parametreyle
              cagir, yoksa numaralar kayar.
        Donen: yeni id listesi, orn [503, 504, 509].

        Id semasi: base = 1000 + cluster_id*100. Ornek c5 -> 1500,1501...
        Normal kume id'leri her zaman < 1000, alt-kumeler her zaman >= 1000,
        boylece cakisma imkansiz (cluster_id=0 dahil). fmt_id bunu '5;3' gosterir.
        """
        base = 1000 + (999 if cluster_id == -1 else cluster_id) * 100
        mask = (self.m.cluster == cluster_id).values
        idx = self.m.index[mask]
        sub_v = self.v[mask]
        if len(sub_v) < min_cluster_size * 2:
            print("cok kucuk, split iptal")
            return []

        clu = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=n_components,
                        metric="cosine", random_state=42).fit_transform(sub_v)
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=min_samples).fit_predict(clu)

        keep = set(keep)
        for orig_i, lab in zip(idx, labels):
            self.m.at[orig_i, "cluster"] = (base + lab) if lab in keep else -1

        label_set = set(labels)
        kept = sorted(base + l for l in keep if l in label_set)
        kept_occ = sum(1 for lab in labels if lab in keep)
        dropped = int(mask.sum()) - kept_occ
        print(f"c{cluster_id} ({int(mask.sum())} occ) -> tutulan: "
              f"{[self.fmt_id(k) for k in kept]} ({kept_occ} occ), "
              f"noise'a giden: {dropped} occ")
        if resummarize:
            self._summarize(ranking="both", verbose=False)
            # yeni alt-kumeleri fit formatinda goster
            new = self.out[self.out.cluster.isin(kept)]
            for _, r in new.iterrows():
                head = (f"c{self.fmt_id(r['cluster'])} n={r['n']:>4} "
                        f"f1={r['lift1']:.1f} f2={r['lift2']:.1f} f3={r['lift3']:.1f} "
                        f"[{r['dom_phase']} {r['dom_lift']:.1f}]")
                print(head)
                if r.get('freq_words'):
                    print(f"   freq: {r['freq_words']}")
                if r.get('ctfidf_words'):
                    print(f"   ctfidf: {r['ctfidf_words']}")
                print(f"   verbs: {r['verb_words']}")
        return kept

    # ----------------------------------------------------------------
    # KUME ICI KWIC   (co-occurrence -> articulation dogrulama)
    # ----------------------------------------------------------------
    def kwic_in_cluster(self, cluster, keyword=None, context_words=None,
                    mode="root", root_len=3, phase=None, venue=None,
                    max_rows=None, newest_first=True):
        """Kumenin occurrence'lari icin KWIC, SADECE metadata context uzerinden.
        keyword: surface'i bu kelimeye uyan occurrence'lari filtreler.
        context_words: context'inde bu kelimeler de gecen occurrence'lari filtreler (AND).
        venue: tek venue (str) ya da venue listesi.
        """
        self._ensure_fit()

        sub = self.m[self.m.cluster == cluster].copy()

        # faz filtresi
        if phase is not None:
            if isinstance(phase, str):
                phase = int(phase.lstrip("fF"))
            sub = sub[sub["phase"] == phase]

        # venue filtresi (tek deger ya da liste)
        if venue is not None:
            venues = [venue] if isinstance(venue, str) else list(venue)
            sub = sub[sub["venue"].isin(venues)]

        # keyword filtresi (surface uzerinden)
        if keyword:
            kw = clean_context(keyword)
            def _kw_match(s):
                # surface coklu form tutabilir (birlesmis occurrence'lar): her
                # mention'i ayri esle, herhangi biri uyarsa occurrence gecer.
                parts = [p.strip(".,;:!?\"'()[] ") for p in clean_context(str(s)).split("|")]
                if mode == "exact":
                    return any(t == kw for t in parts)
                rl = min(root_len, len(kw))
                return any(t[:rl] == kw[:rl] for t in parts)
            sub = sub[sub["surface"].apply(_kw_match)]

        # context_words filtresi (AND)
        if context_words:
            cwords = [clean_context(w) for w in context_words]
            def _has_cwords(ctx):
                toks = [clean_context(t).strip(".,;:!?\"'()[]") for t in str(ctx).split()]
                for cw in cwords:
                    rl = min(root_len, len(cw)) if mode != "exact" else len(cw)
                    if not any((t == cw) if mode == "exact" else (t[:rl] == cw[:rl])
                            for t in toks):
                        return False
                return True
            sub = sub[sub["context"].apply(_has_cwords)]

        # tarih sirala
        if newest_first and "date" in sub.columns:
            sub = sub.sort_values("date", ascending=False)

        print(f"'{clean_context(keyword or self.term)}' (c{cluster}): {len(sub)} occurrence")

        out = sub[["doc_id", "date", "phase", "venue", "context"]].rename(columns={"context": "match"})
        if max_rows:
            out = out.head(max_rows)
        return out
    # ----------------------------------------------------------------
    # CONTEXTUAL DRIFT
    # ----------------------------------------------------------------
    def _early_centroid(self, early_cutoff="2018-01-01"):
        mask = (self.m["date"] < pd.Timestamp(early_cutoff)).values
        if mask.sum() == 0:
            raise ValueError(f"erken pencerede ({early_cutoff} oncesi) chunk yok")
        return self.v[mask].mean(axis=0, keepdims=True), mask.sum()

    def drift_timeline(self, early_cutoff="2018-01-01", freq="2W", min_n=5,
                       save_dir="/Users/erenisikavci/Desktop/Thesis/plots",
                       show=True):
        """Erken donem centroid'ine Ledoit-Wolf Mahalanobis uzakligi."""
        from scipy.spatial.distance import cdist
        from sklearn.covariance import LedoitWolf
    
        mask = (self.m["date"] < pd.Timestamp(early_cutoff)).values
        early = self.v[mask]
        mu = early.mean(0, keepdims=True)
        VI = LedoitWolf().fit(early).precision_
        dist = cdist(self.v, mu, metric="mahalanobis", VI=VI).flatten()
    
        df = pd.DataFrame({"date": self.m["date"], "distance": dist}).sort_values("date")
        df["bin"] = df["date"].dt.to_period(freq)
        grp = df.groupby("bin")["distance"]
        agg = grp.mean().reset_index()
        agg["n"] = grp.size().values
        agg = agg[agg["n"] >= min_n]
        agg["date"] = agg["bin"].dt.to_timestamp()
    
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.scatter(df["date"], df["distance"], alpha=0.15, s=8, color="steelblue", label="chunk")
        ax.plot(agg["date"], agg["distance"], color="crimson", linewidth=2, label=f"{freq} averaged")
        for cut, _ in PHASE_BREAKS:
            ax.axvline(pd.Timestamp(cut), color="gray", linestyle="--", alpha=0.5)
        ax.axvline(pd.Timestamp("2016-07-15"), color="black", linestyle="--", alpha=0.6)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Mahalanobis distance from early centroid", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = f"{save_dir}/drift_timeline_{self.term}.png"
            plt.savefig(path, dpi=600, bbox_inches="tight")
            print("Saved:", path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return agg
    
    
    # ----------------- ARTICULATION CHANGE (JSD)  -----------------
    # APD/dispersion ham embedding GEOMETRISI; bu olcum ise articulation'lar
    # ((meta-)cluster'lar) ARASI YENIDEN DAGILIM uzerinden -> RQ ile birebir,
    # idiolect'e sub-cluster seviyesinde dayanikli, sinirli [0,1], ayristirilabilir,
    # ve phase-lift heatmap ile ayni veriden (iki cozunurluk). self'in kendi fit'i
    # ve cluster_groups'u kullanilir -> export/yeniden-fit yok, cluster-id sorunu yok.


    # Okabe-Ito renk-koru palet (mover cizgileri icin)


    def top_clusters_in_quarter(self, quarter, top=5, n_words=15, verbose=True):
        """Bir CEYREGIN (APD ile ayni birim, 3 aylik) en yogun HAM HDBSCAN kumeleri.
        APD grafigi ceyreklik; orada bir zirve/cukur gordugunde o ceyregi buraya
        ver, en cok occurrence tutan kumeleri gor, kwic_in_cluster ile git oku.

        quarter: '2024Q1' gibi pandas Period string'i, ya da o ceyrege dusen
            herhangi bir tarih ('2024-02-15'); ikisi de o ceyrege cevrilir.
        top: kac kume. n_words: her kume icin kac ctfidf kelimesi (skor sirali).
        Doner: DataFrame (cluster, n, share, ctfidf_words).
        """
        import re as _re
        per = pd.Period(quarter, freq="Q")
        d = self.m["date"]
        mask = (d.dt.to_period("Q") == per).values
        labs = self.m.loc[mask, "cluster"]
        labs = labs[labs != -1]                       # noise'i at
        vc = labs.value_counts().head(top)
        total = int(mask.sum())
        out = pd.DataFrame({"cluster": vc.index.astype(int), "n": vc.values})
        out["share"] = (out["n"] / total).round(3)

        # ctfidf kelimeleri: self.out'tan (zaten ctfidf-secili), sayilari at
        def _clean_words(s):
            if not isinstance(s, str):
                return ""
            # "kelime(345; 148)" / "kelime(345)" -> "kelime", ilk n_words
            ws = [_re.sub(r"\(.*?\)", "", w).strip().lstrip("·")
                for w in s.split(",")]
            ws = [w for w in ws if w]
            return ", ".join(ws[:n_words])
        wmap = {}
        if hasattr(self, "out") and self.out is not None:
            wmap = dict(zip(self.out["cluster"], self.out.get("ctfidf_words", "")))
        out["ctfidf_words"] = out["cluster"].map(lambda c: _clean_words(wmap.get(c, "")))

        if verbose:
            print(f"{per}  ({total} occurrence, noise haric)")
            for _, r in out.iterrows():
                print(f"   c{int(r['cluster']):<4} n={int(r['n']):3d}  {r['share']:.1%}   {r['ctfidf_words']}")
        return out.reset_index(drop=True)
    
