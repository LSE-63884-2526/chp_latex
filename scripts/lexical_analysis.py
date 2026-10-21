"""
Faz-Bazlı Leksik + Semantik Triangulation
==========================================
Amaç: küme/embedding analizinden gelen bulguları (ekonomik-talep register kayması,
altı-ok / halkçılık kelimelerinin faz davranışı) ham lemma korpusundan DOĞRULAMAK.

Üç katman:
  1) phase_frequency  -> kelime SETİNİN üç fazdaki normalize sıklığı (saf frekans)
  2) phase_ppmi       -> bir signifier'ın komşularının FAZ bazlı PPMI'ı (ilişki)
  3) triangulate      -> küme komşuları vs PPMI komşuları örtüşmesi (yöntem-arası doğrulama)

PPMI mantığı ve filtre eşikleri ppmi_dislocation.ipynb ile aynı tutulmuştur.
Korpus: text_lemma (boşlukla tokenize), date kolonu.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from math import log2
import re


# ============================================================
# AYARLAR  (PPMI dosyasıyla tutarlı)
# ============================================================
CORPUS_PATH = "/Users/erenisikavci/Desktop/Thesis/Data/corpus_main_processed.csv"
TEXT_COL = "text_lemma"
DATE_COL = "date"

WINDOW   = 10   # ±N kelime komşuluk penceresi
MIN_COOC = 8    # komşu ego ile pencere içinde en az bu kadar co-occur
MIN_FREQ = 10   # komşu o fazda global en az bu kadar geçsin
MIN_DOCS = 3    # komşu en az bu kadar AYRI konuşmada ego ile geçsin

# ============================================================
# FAZ TANIMI  (memory'deki sınırlar)
#   KK-Early : 2015-01-01  ..  2019-04-01
#   KK-Late  : 2019-04-01  ..  2023-11-08
#   Özel     : 2023-11-08  ..  bugün
# ============================================================
PHASE_BOUNDS = [
    ("KK-Early", pd.Timestamp("2015-01-01"), pd.Timestamp("2019-04-01")),
    ("KK-Late",  pd.Timestamp("2019-04-01"), pd.Timestamp("2023-11-08")),
    ("Ozel",     pd.Timestamp("2023-11-08"), pd.Timestamp("2100-01-01")),
]
PHASES = [p[0] for p in PHASE_BOUNDS]


def assign_phase(dt):
    for name, start, end in PHASE_BOUNDS:
        if start <= dt < end:
            return name
    return None


# ============================================================
# KORPUS YÜKLE + FAZ KOLONU EKLE
# ============================================================
def load_corpus(path=CORPUS_PATH, date_format=None):
    """Korpusu yükler, date'i parse eder, phase kolonu ekler.
    date_format: PPMI dosyasında '%d.%m.%Y' kullanılmış; None ise otomatik dener."""
    df = pd.read_csv(path)
    if date_format:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=date_format, errors="coerce")
    else:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TEXT_COL]).reset_index(drop=True)
    df["phase"] = df[DATE_COL].apply(assign_phase)
    df = df.dropna(subset=["phase"]).reset_index(drop=True)
    return df

# ============================================================
# ALTI OK LEKSİK TARAMA  (ham 'text' kolonu, regex; lemma değil)
# ============================================================
def scan_sixarrow(corpus_df, patterns=None, text_col="text", phase_col="phase"):
    """Lexical triangulation: Alti Ok (ve turevleri) icin korpus-genelinde
    faz-oranli frekans taramasi. Embedding-tabanli kumelemeden bagimsiz bir
    kanit kaynagi.

    corpus_df : ham korpus (full), signifier occurrence'lari degil
    patterns  : {isim: regex}; None ise varsayilan Alti Ok + turevleri
    Doner     : DataFrame (terim x faz sayilari ve % oranlari)
    """
    import pandas as pd

    if patterns is None:
        patterns = {
            "Milliyetçilik":   r"milliyetçi(?:li[kğ])?",
            "Devletçilik":     r"devletçi(?:li[kğ])?",
            "Cumhuriyetçilik": r"cumhuriyetçi(?:li[kğ])?",
            "Halkçılık":       r"halkçı(?:lı[kğ])?",
            "Laiklik":         r"laik(?:li[kğ])?",
            "İnkılap":         r"[iı]nkıla[pb](?:çılı[kğ])?",
            "İhtilal":         r"ihtilal",
            "Devrim":          r"devrim(?:ci(?:li[kğ])?)?",
        }

    phase_totals = corpus_df.groupby(phase_col).size()
    rows = []
    for name, pat in patterns.items():
        mask = corpus_df[text_col].str.contains(pat, case=False, na=False, regex=True)
        row = {"terim": name, "toplam": int(mask.sum())}
        for ph in sorted(phase_totals.index):
            n = int((mask & (corpus_df[phase_col] == ph)).sum())
            row[f"f{ph}_n"] = n
            row[f"f{ph}_%"] = round(n / phase_totals[ph] * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("terim")


# ============================================================
# DISLOKASYON KEYNESS (Log-Likelihood, signifier-bagimsiz) — Dunning 1993
# Bir dislokasyon/secim tarihinin ONCE vs SONRA soyleme GIREN (LL+) ve
# soylemden CIKAN (LL-) kelimeleri. Talep/kimlik "secim oncesi operasyonu,
# sonrasi dususu" = CIKAN (LL-) taraf. Signifier-kosullu DEGIL.
# TEK, KENDINE-YETER fonksiyon: kendi korpus + stopword yukler, tum parametreli.
# ============================================================
def dislocation_keyness(event_date, window_months=4,
                        before_months=None, before_weeks=None,
                        after_months=None, after_weeks=None,
                        corpus=None, corpus_path=CORPUS_PATH,
                        text_col=TEXT_COL, date_col=DATE_COL, date_format=None,
                        speaker=None, speaker_col="speaker",
                        min_freq=5, min_docs=5, min_word_len=3, top_n=40,
                        stop_path="/Users/erenisikavci/Desktop/Thesis/stopwords_tr.txt",
                        verbose=True):
    """Bir dislokasyon/secim tarihinin ONCE (-window_months) vs SONRA
    (+window_months) pencerelerinde Dunning (1993) log-likelihood keyness.
    Signifier-BAGIMSIZ: tum korpusta hangi kelimeler soyleme GIRDI / CIKTI.

    Parametreler
    ------------
    event_date    : 'YYYY-MM-DD' ya da Timestamp (dislokasyon/secim tarihi)
    window_months : olay oncesi/sonrasi pencere yaricapi (ay)
    corpus        : hazir DataFrame (verilirse corpus_path yok sayilir)
    corpus_path   : DataFrame verilmezse buradan yuklenir
    text_col      : lemma/temiz metin kolonu ; date_col: tarih kolonu
    date_format   : tarih parse formati (None -> otomatik ; ham CSV icin '%d.%m.%Y')
    speaker       : verilirse sadece bu konusmaci (speaker_col uzerinden)
    min_freq      : kelime toplam (before+after) en az bu kadar gecsin
    min_docs      : kelime en az bir donemde bu kadar FARKLI konusmada gecsin
                    (tek-miting / tek-olay artefaktini eler)
    min_word_len  : minimum kelime uzunlugu ; top_n: her yonde kac kelime
    stop_path     : stopword dosyasi

    Doner: (entered, left, full)
        entered = soyleme GIREN (LL+, en buyuk)
        left    = soylemden CIKAN (LL-, en kucuk ; talep/kimlik dususu burada)
        full    = tum skorlanmis kelimeler (word, before, after,
                  docs_before, docs_after, ll)
    """
    import re
    from collections import Counter
    import numpy as np
    import pandas as pd

    word_rx = re.compile(r"\b[a-zçğıöşü_]+\b")
    stop = set(w.strip() for w in open(stop_path, encoding="utf-8") if w.strip())

    # --- korpus + pencereler ---
    df = corpus.copy() if corpus is not None else pd.read_csv(corpus_path)
    df[date_col] = (pd.to_datetime(df[date_col], format=date_format, errors="coerce")
                    if date_format else pd.to_datetime(df[date_col], errors="coerce"))
    df = df.dropna(subset=[date_col, text_col])
    if speaker:
        df = df[df[speaker_col] == speaker]
    ev = pd.Timestamp(event_date)
    # asimetrik pencere destegi: before_/after_ verilmezse window_months (simetrik)
    def _off(mo, wk):
        return (pd.DateOffset(months=window_months) if (mo is None and wk is None)
                else pd.DateOffset(months=mo or 0, weeks=wk or 0))
    b0 = ev - _off(before_months, before_weeks)
    a1 = ev + _off(after_months, after_weeks)
    before = df[(df[date_col] >= b0) & (df[date_col] < ev)]
    after  = df[(df[date_col] >= ev) & (df[date_col] < a1)]

    # --- frekans + document-frequency ---
    def counts(texts):
        c, dfreq = Counter(), Counter()
        for t in texts:
            seen = set()
            for w in word_rx.findall(str(t).lower()):
                if len(w) >= min_word_len and w not in stop:
                    c[w] += 1
                    seen.add(w)
            for w in seen:
                dfreq[w] += 1
        return c, dfreq

    cb, dfb = counts(before[text_col])
    ca, dfa = counts(after[text_col])
    Nb, Na = sum(cb.values()), sum(ca.values())
    if Nb == 0 or Na == 0:
        raise ValueError(f"Bos pencere: before={len(before)} sp, after={len(after)} sp.")

    # --- signed Dunning log-likelihood (+ girdi, - cikti) ---
    def ll(a, b):
        e_a, e_b = Na * (a + b) / (Na + Nb), Nb * (a + b) / (Na + Nb)
        val = 0.0
        if a > 0:
            val += a * np.log(a / e_a)
        if b > 0:
            val += b * np.log(b / e_b)
        val *= 2
        return (1 if (a / Na) > (b / Nb) else -1) * val

    rows = []
    for w in set(cb) | set(ca):
        a, b = ca.get(w, 0), cb.get(w, 0)
        if a + b < min_freq:
            continue
        da, db = dfa.get(w, 0), dfb.get(w, 0)
        if max(da, db) < min_docs:
            continue
        lr = np.log2(((a + 0.5) / Na) / ((b + 0.5) / Nb))   # boyuttan-bagimsiz etki buyuklugu (Hardie 2014)
        rows.append({"word": w, "before": b, "after": a,
                     "docs_before": db, "docs_after": da,
                     "ll": round(ll(a, b), 1), "log_ratio": round(lr, 2)})
    full = pd.DataFrame(rows).sort_values("ll", ascending=False).reset_index(drop=True)
    entered = full[full["docs_after"] >= min_docs].nlargest(top_n, "ll").reset_index(drop=True)
    left    = full[full["docs_before"] >= min_docs].nsmallest(top_n, "ll").reset_index(drop=True)

    if verbose:
        print(f"[{ev.date()}] before {b0.date()}→{ev.date()} ({len(before)} sp/{Nb} tok)"
              f" | after {ev.date()}→{a1.date()} ({len(after)} sp/{Na} tok)")
        for title, sub in [
            ("DİSLOKASYON SONRASI SÖYLEME GİREN kelimeler (keyness+)", entered),
            ("DİSLOKASYON ile SÖYLEMDEN ÇIKAN kelimeler (keyness-)", left),
        ]:
            print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
            for _, r in sub.iterrows():
                print(f"  {r['word']:20} LL={r['ll']:7.1f}  LR={r['log_ratio']:+5.1f}  "
                      f"(freq {int(r['before'])}→{int(r['after'])}, "
                      f"docs {int(r['docs_before'])}→{int(r['docs_after'])})")
    return entered, left, full


# ============================================================
# EGO COLLOCATE KEYNESS (Log-Likelihood) — signifier-KOSULLU
# Bir ego'nun (or. "helalleş") +/-window collocate'larini Dunning LL ile skorlar:
# ego-penceresinde korpus tabanina gore ASIRI-temsil edilen kelimeler = ego'nun
# ayirt-edici baglami (chain). PPMI'nin LL muadili -> nadir-kelime gurultusu az.
# event_date verilirse ONCE/SONRA ayri + Jaccard (helalleşme tarzi baglam degisimi).
# ============================================================
def ego_keyness(ego, event_date=None, window_months=4, window=12,
                corpus=None, corpus_path=CORPUS_PATH, text_col=TEXT_COL,
                date_col=DATE_COL, date_format=None, speaker=None, speaker_col="speaker",
                min_cooc=5, min_docs=5, top_n=40,
                stop_path="/Users/erenisikavci/Desktop/Thesis/stopwords_tr.txt",
                verbose=True):
    """Ego collocate keyness (Dunning LL). ego: kok/prefix (re.match ile eslesir).
    event_date None -> tek donem (tum korpus / speaker) collocate keyness, DataFrame.
    event_date verilir -> before/after ayri collocate keyness + Jaccard + new_after
        (helalleşme 'baglam degisimi' analizinin LL hali). Doner dict."""
    import re
    from collections import Counter
    import numpy as np
    import pandas as pd

    word_rx = re.compile(r"\b[a-zçğıöşü_]+\b")
    ego_rx = re.compile(ego)
    stop = set(w.strip() for w in open(stop_path, encoding="utf-8") if w.strip())

    df = corpus.copy() if corpus is not None else pd.read_csv(corpus_path)
    df[date_col] = (pd.to_datetime(df[date_col], format=date_format, errors="coerce")
                    if date_format else pd.to_datetime(df[date_col], errors="coerce"))
    df = df.dropna(subset=[date_col, text_col])
    if speaker:
        df = df[df[speaker_col] == speaker]

    def collocates(texts):
        """Doner: co (ego-penceresi collocate frekansi), cdoc (collocate doc-freq),
        f (donem geneli kelime frekansi), n_win (toplam pencere slotu), N (toplam
        icerik token), ego_occ."""
        co, cdoc, f = Counter(), Counter(), Counter()
        n_win = N = ego_occ = 0
        for t in texts:
            toks = [w for w in word_rx.findall(str(t).lower())]
            for w in toks:
                if len(w) > 2 and w not in stop:
                    f[w] += 1; N += 1
            pos = [i for i, w in enumerate(toks) if ego_rx.match(w)]
            seen = set()
            for i in pos:
                ego_occ += 1
                lo, hi = max(0, i - window), min(len(toks), i + window + 1)
                for j in range(lo, hi):
                    w = toks[j]
                    if j == i or ego_rx.match(w) or len(w) <= 2 or w in stop:
                        continue
                    co[w] += 1; n_win += 1; seen.add((w, id(t)))
            for w, _ in seen:
                cdoc[w] += 1
        return co, cdoc, f, n_win, N, ego_occ

    def score(texts):
        co, cdoc, f, n_win, N, ego_occ = collocates(texts)
        rows = []
        for w, o_in in co.items():
            if o_in < min_cooc or cdoc.get(w, 0) < min_docs:
                continue
            ft, o_out, n_out = f[w], f[w] - o_in, N - n_win
            e_in = n_win * ft / N if N else 0
            ll = 0.0
            if o_in > 0 and e_in > 0:
                ll += o_in * np.log(o_in / e_in)
            if o_out > 0 and (n_out * ft / N) > 0:
                ll += o_out * np.log(o_out / (n_out * ft / N))
            ll *= 2
            if (o_in / n_win) < (ft / N):    # sadece asiri-temsil (pozitif) collocate
                continue
            lr = np.log2((o_in / n_win) / (ft / N))   # PMI-benzeri, boyuttan-bagimsiz etki
            rows.append({"word": w, "cooc": o_in, "docs": cdoc[w],
                         "ll": round(ll, 1), "log_ratio": round(lr, 2)})
        out = pd.DataFrame(rows).sort_values("ll", ascending=False).reset_index(drop=True)
        return out, ego_occ

    if event_date is None:
        tab, ego_occ = score(df[text_col])
        if verbose:
            print(f"[{ego}] tek donem | ego occ={ego_occ} | collocate keyness (LL):")
            for _, r in tab.head(top_n).iterrows():
                print(f"  {r['word']:20} LL={r['ll']:7.1f}  LR={r['log_ratio']:+5.1f}  (cooc {int(r['cooc'])}, docs {int(r['docs'])})")
        return tab.head(top_n)

    ev = pd.Timestamp(event_date)
    b0, a1 = ev - pd.DateOffset(months=window_months), ev + pd.DateOffset(months=window_months)
    before = df[(df[date_col] >= b0) & (df[date_col] < ev)]
    after  = df[(df[date_col] >= ev) & (df[date_col] < a1)]
    tab_b, eb = score(before[text_col])
    tab_a, ea = score(after[text_col])
    top_b = set(tab_b.head(top_n)["word"]); top_a = set(tab_a.head(top_n)["word"])
    jac = len(top_b & top_a) / len(top_b | top_a) if (top_b | top_a) else float("nan")
    new_after = tab_a[~tab_a["word"].isin(top_b)].head(top_n).reset_index(drop=True)
    if verbose:
        print(f"[{ego}] {ev.date()} ±{window_months}mo | ego occ {eb}->{ea} | "
              f"Jaccard(top{top_n})={round(jac, 3)}")
        print(f"\n{'=' * 70}\nEGO YENI BAGLAMI (after, LL collocate keyness)\n{'=' * 70}")
        for _, r in new_after.head(top_n).iterrows():
            print(f"  {r['word']:20} LL={r['ll']:7.1f}  LR={r['log_ratio']:+5.1f}  (cooc {int(r['cooc'])}, docs {int(r['docs'])})")
    return {"before": tab_b, "after": tab_a, "new_after": new_after,
            "jaccard": round(jac, 3), "ego_before": eb, "ego_after": ea}


# ============================================================
# KWIC — Keyword in Context (HAM metin, text_raw)
# Bir tarih araliginda bir kelimenin (kok/prefix) gectigi yerleri +/- window
# kelimelik baglamla, hizali konkordans satirlari olarak gosterir. Keyness
# bulgularini yakin-okuma ile desteklemek icin.
# ============================================================
def kwic(word, start, end, window=8, corpus=None, corpus_path=CORPUS_PATH,
         text_col="text_raw", date_col=DATE_COL, date_format=None,
         speaker=None, speaker_col="speaker", max_hits=50, verbose=True):
    """Keyword-in-context konkordans (HAM metin).
    word    : kok/prefix (re.match ile, Turkce-lower token'a uygulanir)
    start,end: 'YYYY-MM-DD' tarih araligi [start, end)
    window  : keyword'un HER yaninda kac kelime baglam
    text_col: varsayilan 'text_raw' (lemma DEGIL)
    max_hits: en fazla kac satir (sik kelimelerde tasmayi onler; None=hepsi)
    Doner: DataFrame(date, venue, left, keyword, right)."""
    import re
    import pandas as pd

    def tr_lower(s):
        return (s.replace("İ", "i").replace("I", "ı").replace("Ş", "ş")
                 .replace("Ç", "ç").replace("Ö", "ö").replace("Ü", "ü")
                 .replace("Ğ", "ğ").lower())

    df = corpus.copy() if corpus is not None else pd.read_csv(corpus_path)
    df[date_col] = (pd.to_datetime(df[date_col], format=date_format, errors="coerce")
                    if date_format else pd.to_datetime(df[date_col], errors="coerce"))
    df = df.dropna(subset=[date_col, text_col])
    if speaker:
        df = df[df[speaker_col] == speaker]
    df = df[(df[date_col] >= pd.Timestamp(start)) & (df[date_col] < pd.Timestamp(end))]
    df = df.sort_values(date_col)

    # phrase destegi: 'kul_hakkı' / 'asgari ücret' -> ardisik token'lar (her biri prefix)
    subs = [s for s in re.split(r"[_\s]+", tr_lower(word).strip()) if s]
    sub_rx = [re.compile(s) for s in subs]
    k = len(sub_rx)
    tok_rx = re.compile(r"\w+", re.UNICODE)
    rows = []
    for _, r in df.iterrows():
        toks = tok_rx.findall(str(r[text_col]))
        low = [tr_lower(t) for t in toks]
        for i in range(len(low) - k + 1):
            if all(sub_rx[j].match(low[i + j]) for j in range(k)):
                rows.append({"date": r[date_col].date(),
                             "venue": r.get("venue", ""),
                             "left": " ".join(toks[max(0, i - window):i]),
                             "keyword": " ".join(toks[i:i + k]),
                             "right": " ".join(toks[i + k:i + k + window])})
                if max_hits and len(rows) >= max_hits:
                    break
        if max_hits and len(rows) >= max_hits:
            break
    out = pd.DataFrame(rows, columns=["date", "venue", "left", "keyword", "right"])

    if verbose and len(out):
        lw = window * 7
        print(f"KWIC '{word}'  {start} → {end}  ({len(out)} hit"
              + (f", ilk {max_hits}" if max_hits and len(out) == max_hits else "") + ")")
        for _, r in out.iterrows():
            print(f"  {str(r['date'])}  {r['left'][-lw:]:>{lw}}  "
                  f"[ {r['keyword']} ]  {r['right'][:lw]}")
    return out


# ============================================================
# SCRIPT  ->  `python lexical_analysis.py`  tum leksik analizleri terminale basar
#   (ayri notebook gerekmez). Korpus bir kez yuklenir, tum tarihlere gecirilir.
# ============================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    DISLOCATIONS = {
        "Coup 2016":       "2016-07-15",
        "Referendum 2017": "2017-04-16",
        "General 2018":    "2018-06-24",
        "Local 2019":      "2019-03-31",
        "General 2023":    "2023-05-14",
        "Ozel 2023":       "2023-08-11",
        "Local 2024":      "2024-03-31",
        "İmamoğlu 2025":   "2025-03-18",
    }
    _df = pd.read_csv(CORPUS_PATH)   # korpusu bir kez yukle, her cagriya gecir

    print("#" * 72)
    print("# DISLOCATION KEYNESS (Dunning LL) — her tarih: GIREN (+) / CIKAN (-)")
    print("#" * 72)
    for _name, _date in DISLOCATIONS.items():
        print(f"\n\n{'#' * 30} {_name} ({_date}) {'#' * 30}")
        dislocation_keyness(_date, corpus=_df)

    print("\n\n" + "#" * 72)
    print("# EGO KEYNESS — helalleşme baglam degisimi (LL collocate keyness)")
    print("#" * 72)
    ego_keyness("helalleş", event_date="2021-11-01", window_months=16, corpus=_df)


# ============================================================
# REGISTER ANALYSIS
# Tracks thematic term groups over time and around dislocations:
# which register an event displaces, and how long it takes to return.
#
# Patterns in default_stems.py were written for DISPLAY, not counting.
# Used raw they can be wrong: \blaz[^\W_]* matches "lazım" (99% of hits).
# Always run register_contribution() first and read the dominance flag.
# ============================================================

REGISTER_WINDOW_MONTHS = 3       # +- window around an event
REGISTER_BASELINE_MONTHS = 12    # pre-event baseline
REGISTER_MIN_TOKENS = 3000       # skip thinner months
REGISTER_RECOVERY_FRAC = 0.90    # "recovered" = back to this share of baseline
REGISTER_DOMINANCE_MAX = 0.30    # pool unreportable if one term exceeds this

TERM_GROUPS = {
    "economic_demands": [
        r"\baltın[^\W_]*", r"\bbütçe[^\W_]*", r"\bdolar[^\W_]*", r"\bdöviz[^\W_]*",
        r"\bekmek[^\W_]*", r"\bfiyat[^\W_]*", r"\bgeçim[^\W_]*", r"\bişsiz[^\W_]*",
        r"\bkonut[^\W_]*", r"\bkuruş[^\W_]*", r"\blira[^\W_]*", r"\bmaaş[^\W_]*",
        r"\bmaden[^\W_]*", r"\bpara(?!f)[^\W_]*", r"\bprim[^\W_]*", r"\bvergi[^\W_]*",
        r"\bzam(?!an)[^\W_]*",
        r"\bücret(?!siz)[^\W_]*",   # covers "asgari ücret"; `asgari` omitted to avoid double counting
    ],
    "labour_register": [
        r"\bemekli[^\W_]*", r"\bemekçi[^\W_]*", r"\bişçi[^\W_]*", r"\bmemur[^\W_]*",
        r"\bgrev[^\W_]*", r"\bsendika[^\W_]*", r"\btaşeron[^\W_]*", 
    ],
    "moral_register": [          
        r"\bfakir[^\W_]*", r"\bfukara[^\W_]*", r"\byetim[^\W_]*", r"\bmazlum[^\W_]*", r"\bgünah[^\W_]*",
    ],
    "türk": [r"\btürk(?:iye|istan|eş|an)"],
    "communal_identities": [
        r"\bkürt(?!aj)[^\W_]*", r"\balevi[^\W_]*", r"\bsünni[^\W_]*",
        r"\bçerkez[^\W_]*", r"\bçerkes[^\W_]*",
        r"\blaz(?!ı|kiye)[^\W_]*",   # excludes lazım*, Lazkiye
        # `türk` omitted: flat across phases and 42% of the pool, masks the rest
    ],
    "ideological_identities": [
        r"\bmilliyetçi[^\W_]*", r"\bülkücü[^\W_]*", r"\bsolcu[^\W_]*", r"\bliberal[^\W_]*",
        r"\batatürkçü[^\W_]*", r"\bsosyalist\s+(?:enternasyon[ae]l)[^\W_]*", r"\bmuhafaz[aâ]kar[^\W_]*",
    ],
    "antagonist_general": [
        # r"\berdoğan[^\W_]*",
        # r"\bakp\b", r"\bak\s+parti[a-zçğıöşü]*",     # \bakp\b blocks the surname Akpolat
        #r"\bsaray(?!bosna|köy)[^\W_]*",              # moves opposite to erdoğan, report separately
        #r"\bdarbe[^\W_]*",
        r"\bçete(?!le)[^\W_]*",
        r"\btefe[^\W_]*",
        r"\bmafya[^\W_]*",
        r"\bbaron[^\W_]*", 
        r"\buyuşturucu[^\W_]*",
        r"\brant[^\W_]*", 
        r"\brüşvet[^\W_]*", 
        r"\bisraf[^\W_]*",
    ],
    "antagonist_akponly": [
        r"\berdoğan[^\W_]*",
        r"\bakp\b", r"\bak\s+parti[a-zçğıöşü]*",     # \bakp\b blocks the surname Akpolat
    ],

    # Kadin hak kaydi. "sadece kadin gecti mi" degil, HAK TALEBI olarak gecti mi.
    # Yalin \bkadin 7.775 vukuat verir ama cogu hitap ve anma; ayri seri olarak
    # raporlanmali, bu havuza girmemeli.
    #
    # siddet ve esit BILEREK nitelenmis: yalin halleriyle %52 ve %74 oraninda
    # baska konudan geliyorlar (siyasi siddet, ekonomik esitlik). kota da oyle,
    # yalin hali %60 oraninda kadin disi.
    "womens_rights": [
        r"istanbul sözleşme[^\W_]*",                        # 233
        r"kadın[a-zçğıöşü]*\s+(?:yönelik\s+)?şiddet[^\W_]*",  # 202
        r"kadın[\s-]*erkek\s+eşit[^\W_]*",                  # 188
        r"kadın\s+cinayet[^\W_]*",                          # 93
        r"\btaciz[^\W_]*",                                  # 90
        r"kadın\s+hak[^\W_]*",                              # 70
        r"\b8\s+mart|sekiz\s+mart",                         # 69
        r"\btecavüz[^\W_]*",                                # 63
        r"kadın\s+kota[^\W_]*",                             # 33
        r"\bnafaka[^\W_]*",                                 # 31
        r"toplumsal\s+cinsiyet",                            # 27
        r"kadın\s+istihdam[^\W_]*",                         # 19
        r"\b6284\b",                                        # 11
    ],

    # Yalin kadin frekansi. Hak kaydindan AYRI tutuluyor: biri partinin
    # kadinlardan ne siklikta bahsettigini, digeri ne dille bahsettigini olcer.
    "women_mentions": [
        r"\bkadın[^\W_]*",
    ],

    # Gulen hattinin tersine hareket ediyor (KK 0.08 -> Ozel 0.35), o yuzden
    # ayri havuz. Kucuk: 306 vukuat, mekan duzeltmesi icin ince kalabilir.
    "antagonist_tarikat": [
        r"\btarikat[^\W_]*",
        r"\bşeyh(?!ülislam)[^\W_]*",
        r"\bmenzil[^\W_]*",
    ],

    "foundation": [
    # cumhuriyet BILEREK YOK: havuzun %73,8'ini tek basina yutuyordu.
    # Ayri seri olarak raporlanmali, bu havuza girmemeli.
    r"kurtuluş\s+savaş[^\W_]*",              # 674  (yalin "kurtuluş" degil:
                                             #       "kurtuluş recetesi" vb. var)
    r"\bkurucu[^\W_]*",                      # 762
    r"\b19(?:19|20|21|22|23)\b",             # 479  kurulus yillari
    r"\blozan[^\W_]*",                       # 224
    r"kuva[yi]?[ıi]?\s+milliye",             # 204
    r"19\s+mayıs",                           # 161
    r"sivas\s+kongre[^\W_]*",                # 157  yalin "sivas" miting sehri
    r"mill[iî]\s+mücadele",                  # 154
    r"29\s+ekim",                            # 107
    r"amasya\s+(?:genelge|tamim|görüşme)[^\W_]*",  # 51
    r"erzurum\s+kongre[^\W_]*",              # 44
    r"misak[ıi]?[\s-]milli",                 # 8
    ],


    "antagonist_gulen": [
    r"\bfetö[^\W_]*", r"\bparalel[^\W_]*",
    r"\bgülen(?:ci|ist|\s+cemaat|\s+örgüt)[^\W_]*|\bfethullah[^\W_]*",
    r"\bpensilvanya[^\W_]*|\bpennsylvania[^\W_]*",
    ],
    
    "antagonist_tutelage": [r"\bvesayet[^\W_]*"],
    "antagonist_terror": [r"\bterör[^\W_]*"],
    "coup_frame": [
        # kept out of the antagonist groups: referent inverts (2016 putschists,
        # 2025 the incumbent). Pooling would average the inversion away.
        r"\bdarbe[^\W_]*", r"\bcunta[^\W_]*", r"\bkayyım[^\W_]*", r"\bkayyum[^\W_]*",
    ],
}

DISLOCATIONS = {
    "Coup attempt 2016":     "2016-07-15",
    "Referendum 2017":       "2017-04-16",
    "General election 2018": "2018-06-24",   # crowding-out control: economic event
    "Local election 2019":   "2019-03-31",
    "General election 2023": "2023-05-14",
    "Local election 2024":   "2024-03-31",
    "Bahçeli's call 2024":   "2024-10-22",
    "İmamoğlu arrest 2025":  "2025-03-19",
    "Istanbul trustee 2025":     "2025-09-02",
}

_REG_TOK = re.compile(r"[a-zçğıöşü]+")


def _tr_lower(s):
    """Turkish-aware lowercase."""
    return str(s).replace("İ", "i").replace("I", "ı").lower()


def register_contribution(group, corpus=None, text_col="text_bert",
                          phase_col="phase", verbose=True):
    """Per-term rate (per 10,000 tokens) by phase, plus a dominance check.

    Run this before reporting any pooled series. If one term is more than
    REGISTER_DOMINANCE_MAX of the pool, the pool is really just that term.
    Returns (DataFrame, dominance_ok).
    """
    pats = TERM_GROUPS[group] if isinstance(group, str) else group
    if corpus is None:
        corpus = load_corpus()
    text = {p: _tr_lower(" ".join(g[text_col].astype(str)))
            for p, g in corpus.groupby(phase_col)}
    total = {p: len(_REG_TOK.findall(t)) for p, t in text.items()}
    phases = sorted(text)

    out = pd.DataFrame(
        {pat: [len(re.findall(pat, text[p])) / total[p] * 10000 for p in phases]
         for pat in pats},
        index=phases).T
    out.columns = [f"P{p}" for p in phases]

    last = out.columns[-1]
    share = out[last] / out[last].sum() if out[last].sum() else out[last]
    out["share_last"] = share
    dominance_ok = bool(share.max() <= REGISTER_DOMINANCE_MAX)

    if verbose:
        print(out.round(2).sort_values(last, ascending=False).to_string())
        print(f"\n  largest term = {share.max():.0%} of the pool")
        print("  -> pool reportable" if dominance_ok else
              "  -> DO NOT pool, report term by term")
    return out, dominance_ok


def register_recovery(group, events=None, corpus=None, text_col="text_bert",
                      date_col=DATE_COL, verbose=True):
    """How a register moves around each dislocation, and when it returns.

    Baseline = REGISTER_BASELINE_MONTHS before the event. Change = mean over the
    REGISTER_WINDOW_MONTHS after it. Recovery = months until the series regains
    REGISTER_RECOVERY_FRAC of baseline (searched 24 months out).

    Keep at least one event whose content matches the register (e.g. the Aug-2018
    currency crisis for economic demands): if the register rises there and falls
    elsewhere, the drops are not just agenda crowding-out.
    """
    pats = TERM_GROUPS[group] if isinstance(group, str) else group
    if events is None:
        events = DISLOCATIONS
    if corpus is None:
        corpus = load_corpus()
    df = corpus.dropna(subset=[date_col, text_col]).sort_values(date_col).copy()
    df["_m"] = df[date_col].dt.to_period("M")

    monthly = {}
    for m, g in df.groupby("_m"):
        text = _tr_lower(" ".join(g[text_col].astype(str)))
        n = len(_REG_TOK.findall(text))
        if n < REGISTER_MIN_TOKENS:
            continue
        monthly[m.to_timestamp()] = sum(len(re.findall(p, text)) for p in pats) / n * 10000
    s = pd.Series(monthly).sort_index()
    s = s.rolling(REGISTER_WINDOW_MONTHS, center=True, min_periods=2).mean()

    rows = []
    for label, date in events.items():
        d = pd.Timestamp(date)
        base = s[(s.index >= d - pd.DateOffset(months=REGISTER_BASELINE_MONTHS)) & (s.index < d)]
        post = s[(s.index >= d) & (s.index < d + pd.DateOffset(months=REGISTER_WINDOW_MONTHS))]
        if len(base) < 6 or len(post) < 2:
            rows.append((label, np.nan, np.nan, np.nan, "insufficient data"))
            continue
        b, p = base.mean(), post.mean()
        change = (p - b) / b * 100
        recovery = "no drop"
        if change < -15:
            fwd = s[(s.index >= d) & (s.index <= d + pd.DateOffset(months=24))]
            back = fwd[fwd >= REGISTER_RECOVERY_FRAC * b]
            if len(back):
                recovery = f"{round((back.index[0] - d).days / 30.44):.0f} months"
            else:
                # Censoring guard. Where the corpus ends fewer than 24 months
                # after the event, a missing recovery is UNOBSERVED, not absent.
                # The last three dislocations are all in this position (17, 12
                # and 6 months of data), so ">24 months" there would assert
                # something the data cannot support.
                avail = ((s.index.max().year - d.year) * 12
                         + s.index.max().month - d.month)
                recovery = (">24 months" if avail >= 24
                            else f"censored ({avail}mo)")
        rows.append((label, b, p, change, recovery))

    out = pd.DataFrame(rows, columns=["event", "baseline", "post", "change_pct", "recovery"])
    if verbose:
        print(out.round(1).to_string(index=False))
    return out


# =============================================================================
# EQUIVALENTIAL CHAINS
#
# Laclau's chain of equivalence has a surface signature: the speaker enumerates.
# "Kürt'üyle, Alevi'siyle, Sünni'siyle, hepimiz..." is a chain performed in the
# text, not a distributional similarity inferred from it.
#
# Measuring it here rather than in embedding space is deliberate. The geometric
# route depends on whitening, UMAP and HDBSCAN settings, each of which has to be
# defended. Enumeration depends on almost none of that.
#
# Turkish builds chains with several constructions, and no single one is the
# chain:
#     -iyle repetition   Kürt'üyle Türk'üyle
#     -i de / -si de     Kürt'ü de Alevi'si de
#     hem ... hem        hem işçi hem emekli
#     comma list         işçi, memur, emekli, esnaf
#     bare juxtaposition işçi memur emekli
# Detection is therefore proximity-based (terms of a known class within
# CHAIN_MAX_GAP words), and the connective found between them is recorded rather
# than required. Syntax alone would catch city lists; the term class is what
# makes a run of tokens a chain.
#
# Three measures fall out:
#     rate           how often chains are built
#     length         how many terms are linked
#     heterogeneity  how many DIFFERENT classes a chain draws on
# The third is the one that matters theoretically. Equivalence links differences,
# so a chain spanning ethnicity, class and gender reaches further than one that
# stays inside a single class.
# =============================================================================

CHAIN_MAX_GAP = 4      # words allowed between two chain members
CHAIN_MIN_LEN = 2      # a chain needs at least this many members

# Negative lookaheads are the ones validated against the corpus: `laz` would
# otherwise take lazım, `kız` would take kızgın, `roman` would take romantik.
CHAIN_CLASSES = {
    "etnik": {
        "türk":    r"\btürk(?!iy|ce|çe|men|iş)[^\W_]*",
        "kürt":    r"\bkürt(?!aj)[^\W_]*",
        "laz":     r"\blaz(?!ı|kiye|er)[^\W_]*",
        "çerkes":  r"\bçerke[sz][^\W_]*",
        "arap":    r"\barap[^\W_]*",
        "roman":   r"\broman(?!t|ı)[^\W_]*",
        "yörük":   r"\byörük[^\W_]*",
        "arnavut": r"\barnavut[^\W_]*",
        "ermeni":  r"\bermeni[^\W_]*",
        "rum":     r"\brum(?!e|u)[^\W_]*",
        "boşnak":  r"\bboşnak[^\W_]*",
    },
    # Sub-sects (hanefi, şafi, caferi, nusayri) are absent from this corpus:
    # under 10 occurrences each. Political speech stops at alevi/sünni.
    "mezhep": {
        "alevi":   r"\balevi[^\W_]*",
        "sünni":   r"\bsünni[^\W_]*",
        "bektaşi": r"\bbektaşi[^\W_]*",
    },
    # One axis with two poles, kept apart from `mezhep`: degree of religiosity
    # is not sect membership. `laik` sits here rather than under `siyasi`
    # because its chain partner is `dindar`.
    "dindarlık": {
        "dindar":     r"\bdindar[^\W_]*",
        "mütedeyyin": r"\bmütedeyyin[^\W_]*",
        "laik":       r"\blaik[^\W_]*",
    },
    "demografik": {
        "kadın":   r"\bkadın[^\W_]*",
        "erkek":   r"\berkek[^\W_]*",
        "genç":    r"\bgenç[^\W_]*",
        "yaşlı":   r"\byaşlı[^\W_]*",
        "çocuk":   r"\bçocuk[^\W_]*",
        "anne":    r"\banne[^\W_]*",
        "baba":    r"\bbaba(?!lar\s+günü)[^\W_]*",
        "evlat":   r"\bevlat[^\W_]*",
        "kız":     r"\bkız(?!g|ı[lm]|dır|ar)[^\W_]*",
        "engelli": r"\bengelli[^\W_]*",
    },
    "sınıf": {
        "emekli":   r"\bemekli[^\W_]*",
        "çiftçi":   r"\bçiftçi[^\W_]*",
        "işçi":     r"\bişçi[^\W_]*",
        "esnaf":    r"\besnaf[^\W_]*",
        "öğrenci":  r"\böğrenci[^\W_]*",
        "memur":    r"\bmemur[^\W_]*",
        "üretici":  r"\büretici[^\W_]*",
        "emekçi":   r"\bemekçi[^\W_]*",
        "sanayici": r"\bsanayici[^\W_]*",
        "taşeron":  r"\btaşeron[^\W_]*",
        "köylü":    r"\bköylü[^\W_]*",
        "işveren":  r"\bişveren[^\W_]*",
        "madenci":  r"\bmadenci[^\W_]*",
    },
    "siyasi": {
        "demokrat":     r"\bdemokrat(?!ik|ikleş)[^\W_]*",
        "milliyetçi":   r"\bmilliyetçi[^\W_]*",
        "muhafazakâr":  r"\bmuhafazak[âa]r[^\W_]*",
        "sosyalist":    r"\bsosyalist[^\W_]*",
        "ülkücü":       r"\bülkücü[^\W_]*",
        "atatürkçü":    r"\batatürkçü[^\W_]*",
        "solcu":        r"\bsolcu[^\W_]*",
        "liberal":      r"\bliberal[^\W_]*",
        "sağcı":        r"\bsağcı[^\W_]*",
        "devrimci":     r"\bdevrimci[^\W_]*",
        "cumhuriyetçi": r"\bcumhuriyetçi[^\W_]*",
    },
    # The frontier has two sides. Laclau's antagonism requires the opposing pole
    # to be constructed by equivalence as well, so the antagonist terms are a
    # class in their own right and chains can be measured on that side too.
    "antagonist": {
        "fetö":       r"\bfetö[^\W_]*",
        "yolsuz":     r"\byolsuz[^\W_]*",
        "rüşvet":     r"\brüşvet[^\W_]*",
        "tefeci":     r"\btefeci[^\W_]*",
        "uyuşturucu": r"\buyuşturucu[^\W_]*",
        "yandaş":     r"\byandaş[^\W_]*",
        "çete":       r"\bçete(?!le)[^\W_]*",
        "beşli çete": r"beşli\s+çete[^\W_]*",
        "darbeci":    r"\bdarbeci[^\W_]*",
        "hırsız":     r"\bhırsız[^\W_]*",
        "baron":      r"\bbaron[^\W_]*",
        "cemaat":     r"\bcemaat[^\W_]*",
        "mafya":      r"\bmafya[^\W_]*",
        "zalim":      r"\bzalim[^\W_]*",
        "rantçı":     r"\brantçı[^\W_]*|\brantiye[^\W_]*",
    },
}

# Connective found between two members. Recorded, not required: bare
# juxtaposition ("işçi memur emekli") is a chain too.
_CHAIN_MARKERS = [
    ("iyle",  re.compile(r"s?[ıiuü]yl[ae]\b")),
    ("de",    re.compile(r"\bde\b|\bda\b")),
    ("hem",   re.compile(r"\bhem\b")),
    ("comma", re.compile(r",")),
    ("ve",    re.compile(r"\bve\b")),
]

_CHAIN_WORD = re.compile(r"\S+")


def _compile_chain_regex(classes):
    """One alternation over every term, with index-named groups.

    Named groups let re report which term matched via `lastgroup`, which is far
    cheaper than testing each pattern in turn. Group names must be identifiers,
    so terms are addressed by index rather than by their Turkish spelling.
    """
    lookup, parts = [], []
    for cls, terms in classes.items():
        for term, pat in terms.items():
            parts.append(f"(?P<t{len(lookup)}>{pat})")
            lookup.append((cls, term))
    return re.compile("|".join(parts)), lookup


def find_chains(text, classes=None, max_gap=CHAIN_MAX_GAP, min_len=CHAIN_MIN_LEN):
    """Enumerations of class terms in one document.

    Returns a list of dicts: members, classes, length, heterogeneity, markers.
    A run continues while consecutive members are at most `max_gap` words apart.

    Note on length-2 chains: adjective-noun pairs ("genç kadın") are picked up
    alongside genuine two-member chains. Filter on `markers` if that matters,
    or read length>=3 only. Reported rather than silently dropped.
    """
    if classes is None:
        classes = CHAIN_CLASSES
    rx, lookup = _compile_chain_regex(classes)
    tl = _tr_lower(text)

    starts = [m.start() for m in _CHAIN_WORD.finditer(tl)]
    if not starts:
        return []

    def word_index(pos):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    hits = []
    for m in rx.finditer(tl):
        cls, term = lookup[int(m.lastgroup[1:])]
        hits.append((word_index(m.start()), m.start(), m.end(), cls, term))
    if not hits:
        return []
    hits.sort()

    chains, run = [], [hits[0]]
    for h in hits[1:]:
        if h[0] - run[-1][0] <= max_gap and h[0] != run[-1][0]:
            run.append(h)
        else:
            if len(run) >= min_len:
                chains.append(run)
            run = [h]
    if len(run) >= min_len:
        chains.append(run)

    out = []
    for run in chains:
        marks = set()
        for a, b in zip(run, run[1:]):
            gap = tl[a[2]:b[1]]
            for name, mrx in _CHAIN_MARKERS:
                if mrx.search(gap):
                    marks.add(name)

        # Length counts DISTINCT terms. Rhetorical repetition ("memur, memur,
        # memur") is emphasis, not equivalence: a chain links differences, so
        # the same element repeated adds nothing to its reach. The raw run size
        # is kept as n_raw in case the repetition itself is of interest.
        members, seen = [], set()
        for h in run:
            if h[4] not in seen:
                seen.add(h[4])
                members.append((h[4], h[3]))
        if len(members) < min_len:
            continue

        cls_list = [c for _, c in members]
        out.append({
            "members":       [t for t, _ in members],
            "classes":       cls_list,
            "length":        len(members),
            "n_raw":         len(run),
            "heterogeneity": len(set(cls_list)),
            "markers":       sorted(marks),
            "start":         run[0][1],
        })
    return out


def chain_table(corpus=None, text_col="text_bert", date_col=DATE_COL,
                classes=None, min_len=CHAIN_MIN_LEN):
    """Every chain in the corpus as one row, with its document's date.

    This is the base table; the summary helpers below all read from it.
    """
    if corpus is None:
        corpus = load_corpus()
    rows = []
    for _, r in corpus.dropna(subset=[text_col]).iterrows():
        for ch in find_chains(str(r[text_col]), classes=classes, min_len=min_len):
            rows.append({
                "date":          r.get(date_col),
                "length":        ch["length"],
                "n_raw":         ch["n_raw"],
                "heterogeneity": ch["heterogeneity"],
                "members":       " · ".join(ch["members"]),
                "classes":       " · ".join(ch["classes"]),
                "markers":       ",".join(ch["markers"]) or "bare",
            })
    return pd.DataFrame(rows)


def chain_summary(chains=None, corpus=None, verbose=True, **kw):
    """Rate, length distribution and heterogeneity across the whole corpus."""
    if chains is None:
        chains = chain_table(corpus=corpus, **kw)
    if verbose:
        print(f"chains: {len(chains):,}")
        print(f"mean length: {chains.length.mean():.2f}   "
              f"mean heterogeneity: {chains.heterogeneity.mean():.2f}")
        print("\nlength")
        print(chains.length.value_counts().sort_index().to_string())
        print("\nheterogeneity (distinct classes per chain)")
        print(chains.heterogeneity.value_counts().sort_index().to_string())
        print("\nconnective")
        print(chains.markers.value_counts().head(8).to_string())
    return chains


def chain_series(chains=None, corpus=None, freq="Q", text_col="text_bert",
                 date_col=DATE_COL, verbose=True, **kw):
    """Chain rate per 10,000 tokens, plus mean length and heterogeneity.

    The rate is normalised by tokens so it is not driven by how much the leader
    spoke in a given period.
    """
    if corpus is None:
        corpus = load_corpus()
    if chains is None:
        chains = chain_table(corpus=corpus, text_col=text_col,
                             date_col=date_col, **kw)

    tok = (corpus.dropna(subset=[date_col, text_col])
                 .assign(_p=lambda d: d[date_col].dt.to_period(freq))
                 .groupby("_p")[text_col]
                 .apply(lambda g: len(_REG_TOK.findall(_tr_lower(" ".join(g.astype(str)))))))

    c = chains.dropna(subset=["date"]).copy()
    c["_p"] = c["date"].dt.to_period(freq)
    agg = c.groupby("_p").agg(n=("length", "size"),
                              mean_len=("length", "mean"),
                              mean_het=("heterogeneity", "mean"))
    out = agg.join(tok.rename("tokens"), how="inner")
    out["per10k"] = out.n / out.tokens * 10000
    out = out[["n", "per10k", "mean_len", "mean_het"]]
    if verbose:
        print(out.round(2).to_string())
    return out


def chain_composition(chains=None, corpus=None, top=20, verbose=True, **kw):
    """Which terms and which class pairs are linked.

    The class-pair counts answer the question the cluster analysis could not:
    whether the party links across social divisions or stays inside one.
    """
    from itertools import combinations
    if chains is None:
        chains = chain_table(corpus=corpus, **kw)

    terms = Counter()
    pairs = Counter()
    for _, r in chains.iterrows():
        ms = r["members"].split(" · ")
        cs = r["classes"].split(" · ")
        terms.update(ms)
        for a, b in combinations(sorted(set(cs)), 2):
            pairs[(a, b)] += 1

    t = pd.Series(terms).sort_values(ascending=False).head(top)
    p = pd.Series({f"{a} + {b}": n for (a, b), n in pairs.items()}
                  ).sort_values(ascending=False).head(top)
    if verbose:
        print("most linked terms")
        print(t.to_string())
        print("\nmost linked class pairs")
        print(p.to_string())
    return t, p


# =============================================================================
# PLOTS
#
# A report reader takes the shape, not the number. These produce the figure that
# carries a finding on its own.
#
# One deliberate departure from the group-appeals literature: no OLS line across
# the full period. Thau fits one because his series has no structural break. This
# corpus changes leader in November 2023, exactly at the last phase boundary, so
# a single straight line would smooth over a documented discontinuity. A local
# smoother guides the eye without asserting a functional form; if a trend line is
# wanted, `split_at` fits one per side of the break instead.
# =============================================================================

PLOT_INK   = "#111111"
PLOT_MUTED = "#8A939C"
PLOT_GRID  = "#E4E8EC"
PLOT_LINE  = ["#1D4E89", "#B5541A", "#13715B", "#9B2226",
              "#6B2D8B", "#50596B", "#B08900"]

LEADER_CHANGE = "2023-11-08"


def _series_per10k(pats, corpus, freq, text_col, date_col):
    """Rate per 10,000 tokens, aggregated to `freq`."""
    df = corpus.dropna(subset=[date_col, text_col]).copy()
    df["_p"] = df[date_col].dt.to_period(freq)
    rows = {}
    for p, g in df.groupby("_p"):
        text = _tr_lower(" ".join(g[text_col].astype(str)))
        n = len(_REG_TOK.findall(text))
        if n < REGISTER_MIN_TOKENS:
            continue
        rows[p.to_timestamp()] = sum(len(re.findall(x, text)) for x in pats) / n * 10000
    return pd.Series(rows).sort_index()


def register_plot(groups, corpus=None, freq="Q", text_col="text_bert",
                  date_col=DATE_COL, events=None, smooth=0.35,
                  split_at=LEADER_CHANGE, trend=False, separate=False,
                  outfile=None, title=None, ylabel=None, figsize=None):
    """Rate per 10,000 tokens over time for one or more term groups.

    groups   : a TERM_GROUPS key, a list of keys, or {label: patterns}
    freq     : "M", "Q" or "Y". Q is usually right: M is noisy, Y too coarse
               for a local smoother to mean anything.
    smooth   : LOWESS span, or None to plot the raw line only
    split_at : draw the leader change and, with trend=True, fit one line each
               side of it rather than one across
    separate : small multiples instead of one panel. Easier for a reader who is
               comparing shapes rather than levels.
    events   : {label: date} vertical markers, defaults to DISLOCATIONS
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if corpus is None:
        corpus = load_corpus()
    if isinstance(groups, str):
        groups = [groups]
    if isinstance(groups, (list, tuple)):
        groups = {g: TERM_GROUPS[g] for g in groups}

    series = {k: _series_per10k(v, corpus, freq, text_col, date_col)
              for k, v in groups.items()}
    series = {k: s for k, s in series.items() if len(s) > 3}
    if not series:
        raise ValueError("no series long enough to plot")

    n = len(series)
    if separate and n > 1:
        figsize = figsize or (8.6, 2.0 * n + 0.8)
        fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
        axes = list(axes)
    else:
        figsize = figsize or (8.6, 4.2)
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax] * n

    ev = DISLOCATIONS if events is None else events

    for i, ((label, s), ax) in enumerate(zip(series.items(), axes)):
        c = PLOT_LINE[i % len(PLOT_LINE)]

        # events first, so they sit behind the data
        if ev and (i == 0 or separate):
            for _, d in ev.items():
                ax.axvline(pd.Timestamp(d), color=PLOT_GRID, lw=0.9, zorder=0)
        if split_at and (i == 0 or separate):
            ax.axvline(pd.Timestamp(split_at), color=PLOT_MUTED,
                       lw=1.0, ls=(0, (4, 3)), zorder=1)

        ax.plot(s.index, s.values, color=c, lw=0.9, alpha=0.45 if smooth else 1.0,
                marker="o", ms=2.6, mew=0, zorder=2,
                label=label if not separate else None)

        if smooth and len(s) > 6:
            x = mdates.date2num(s.index.to_pydatetime())
            z = lowess(s.values, x, frac=smooth, return_sorted=True)
            ax.plot(mdates.num2date(z[:, 0]), z[:, 1], color=c, lw=2.1, zorder=3)

        if trend and split_at:
            for lo, hi in [(s.index.min(), pd.Timestamp(split_at)),
                           (pd.Timestamp(split_at), s.index.max())]:
                seg = s[(s.index >= lo) & (s.index <= hi)]
                if len(seg) < 4:
                    continue
                x = mdates.date2num(seg.index.to_pydatetime())
                b, a = np.polyfit(x, seg.values, 1)
                ax.plot(mdates.num2date(x), a + b * x, color=PLOT_INK,
                        lw=1.1, ls="--", zorder=4)

        if separate:
            ax.set_title(label, loc="left", fontsize=10, color=PLOT_INK,
                         fontweight="bold", pad=4)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=PLOT_GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PLOT_GRID)
        ax.tick_params(colors=PLOT_MUTED, labelsize=8.5, length=3)

    tail = axes[-1]
    tail.set_xlabel("")
    fig.supylabel(ylabel or "10.000 kelimede", fontsize=9,
                  color=PLOT_MUTED, x=0.015)
    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold",
                     color=PLOT_INK, x=0.015, ha="left", y=0.995)
    if not separate and n > 1:
        axes[0].legend(frameon=False, fontsize=9, loc="upper left",
                       labelcolor=PLOT_INK)
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"saved: {outfile}")
    return fig, series


# =============================================================================
# CHANGE TABLE
#
# Two multi-year windows rather than the three phases: the phase boundary sits
# exactly on the leader change, so a phase comparison reads as a comparison of
# leaders. Windows with a gap between them compare periods instead, and the gap
# keeps the boundary from doing the work.
# =============================================================================

REGISTER_LABELS_TR = {
    "economic_demands":           "Ekonomik talepler",
    "labour_register":            "Emek ve çalışma",
    "moral_register":             "Ahlaki hak ediş",
    "communal_identities":        "Etnik ve inanç kimlikleri",
    "ideological_identities":     "İdeolojik kimlikler",
    "antagonist_general":         "Yolsuzluk ve çete",
    "antagonist_akponly":         "Erdoğan ve AKP",
    "antagonist_religious_order": "Cemaat",
    "antagonist_tutelage":        "Vesayet",
    "antagonist_terror":          "Terör",
    "coup_frame":                 "Darbe çerçevesi",
    "foundation":                 "Kuruluş anlatısı",
    "womens_rights":              "Kadın hakları",
    "women_mentions":             "Kadın (anma sıklığı)",
    "antagonist_gulen":           "Gülen hareketi",
    "antagonist_tarikat":         "Tarikatlar",
}


def register_change(groups=None, corpus=None, text_col="text_bert",
                    date_col=DATE_COL, split_at=LEADER_CHANGE,
                    early=None, late=None, labels=None, names=None,
                    verbose=True):
    """Rate per 10,000 tokens either side of a split, and the change.

    split_at : date. Everything before it is compared with everything after.
               Defaults to LEADER_CHANGE. Set to None and pass early/late as
               ("2015","2019") style year windows to compare periods instead.

               What such a table can and cannot say: the two leaders do not
               overlap in this corpus, so a split at the handover cannot
               separate the leader from the period. Whatever else changed in
               November 2023 sits in the same column. Read it as before and
               after the handover, not as what one leader did. The changepoint
               dates are the check: a register breaking somewhere other than
               2023-11 is not moving with the leader.

    names    : (left, right) column labels. Sensible defaults follow the split.

    Column labels are carried on `.attrs["names"]` for the LaTeX helper.
    """
    if corpus is None:
        corpus = load_corpus()
    if groups is None:
        groups = list(TERM_GROUPS)
    labels = labels or REGISTER_LABELS_TR

    df = corpus.dropna(subset=[date_col, text_col]).copy()

    def tokens(mask):
        t = _tr_lower(" ".join(df.loc[mask, text_col].astype(str)))
        return t, len(_REG_TOK.findall(t))

    if split_at:
        cut = pd.Timestamp(split_at)
        t_e, n_e = tokens(df[date_col] < cut)
        t_l, n_l = tokens(df[date_col] >= cut)
        col_names = names or ("Kılıçdaroğlu", "Özel")
    else:
        y = df[date_col].dt.year
        t_e, n_e = tokens((y >= int(early[0])) & (y <= int(early[1])))
        t_l, n_l = tokens((y >= int(late[0])) & (y <= int(late[1])))
        col_names = names or (f"{early[0]}--{early[1]}", f"{late[0]}--{late[1]}")

    rows = []
    for g in groups:
        pats = TERM_GROUPS[g] if isinstance(g, str) else g
        e = sum(len(re.findall(p, t_e)) for p in pats) / n_e * 10000
        l = sum(len(re.findall(p, t_l)) for p in pats) / n_l * 10000
        # Per cent change explodes on a small base: 0.6 -> 7.6 reads as
        # +1152 per cent while 129.3 -> 133.4 reads as +3, although the two
        # moves are the same size in absolute terms. Fold change keeps the
        # scale readable and comparable across registers, and direction is
        # carried by the arrow rather than by the sign.
        rows.append({"group": g,
                     "label": labels.get(g, g),
                     "early": e, "late": l,
                     "abs": l - e,
                     "fold": (max(e, l) / min(e, l)) if min(e, l) > 0 else np.nan,
                     "change": (l - e) / e * 100 if e else np.nan})

    out = pd.DataFrame(rows).sort_values("change", ascending=False)
    out.attrs["names"] = col_names
    if verbose:
        print(f"{col_names[0]}  vs  {col_names[1]}   (per 10,000 tokens)")
        print(f"  n = {n_e:,} / {n_l:,} tokens\n")
        print(f"  {'':<26}{col_names[0]:>10}{col_names[1]:>10}{'kat':>8}{'fark':>8}")
        for r in out.itertuples():
            ok = "^" if r.abs > 0 else "v" if r.abs < 0 else "-"
            dusuk = "  <- dusuk taban" if min(r.early, r.late) < 1.0 else ""
            print(f"  {r.label:<26}{r.early:>10.1f}{r.late:>10.1f}"
                  f"{r.fold:>7.1f}x{r.abs:>+8.1f} {ok}{dusuk}")
    return out


def register_change_latex(table=None, outfile=None, rise_at=10.0,
                          early_label=None, late_label=None, **kw):
    """LaTeX table, rising registers grouped above the rest.

    `rise_at` is the threshold in per cent for counting as a rise. Anything
    below it is grouped as flat-or-falling, because a report reader should not
    have to decide whether +3 per cent means anything.
    """
    if table is None:
        table = register_change(verbose=False, **kw)
    auto = table.attrs.get("names", ("Önce", "Sonra"))
    early_label = early_label or auto[0]
    late_label = late_label or auto[1]

    up = table[table.change >= rise_at]
    rest = table[table.change < rise_at]

    def block(rows):
        out = []
        for r in rows.itertuples():
            arrow = (r"\artan" if r.change >= rise_at else
                     r"\azalan" if r.change <= -rise_at else r"\duraga")
            out.append(f"  {r.label} & {r.early:.1f} & {r.late:.1f} & "
                       f"{r.change:+.0f}\\% & {arrow} \\\\")
        return "\n".join(out)

    tex = f"""\\begin{{table}}[H]
\\centering
\\caption{{Kayıtların iki dönem arasındaki değişimi}}
\\label{{tab:register-change}}
\\begin{{tabular}}{{@{{}}l r r r c@{{}}}}
\\toprule
 & {early_label} & {late_label} & Değişim & \\\\
\\midrule
\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{Artanlar}}}} \\\\
{block(up)}
\\addlinespace[0.5em]
\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{Azalanlar ve değişmeyenler}}}} \\\\
{block(rest)}
\\bottomrule
\\end{{tabular}}
\\kaynaknot{{Oranlar 10.000 kelimede vukuat sayısıdır, yani konuşma hacminden
bağımsızdır. Tabanı 10.000'de 1'in altında olan satırlarda yüzde oynaktır;
oralarda iki oranı doğrudan karşılaştırın. \\%{rise_at:.0f}'in altındaki
hareketler değişmemiş sayılmıştır.}}
\\end{{table}}
"""
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"saved: {outfile}")
    return tex


# =============================================================================
# CHANGEPOINTS
#
# register_recovery is event-anchored: it asks what happened around dates given
# in advance. That cannot see a change which falls between two events, and a
# register can move for reasons the calendar does not carry.
#
# This asks the opposite question. Where does the series actually change? The
# split point is found by maximising a two-sample t statistic over every
# candidate month, and its p value comes from a block permutation that preserves
# short-run autocorrelation. Across several registers the p values are corrected
# with Benjamini-Hochberg.
#
# `abruptness` guards against reading a slope as a step. A changepoint test
# always returns a date, even when the series drifts. The measure compares how
# much variance a step at that date explains against how much a straight line
# explains: above 1 the break is the better description, below 1 the register is
# drifting and the date is an artefact of forcing a step onto a trend.
# =============================================================================

def register_changepoint(groups=None, corpus=None, text_col="text_bert",
                         date_col=DATE_COL, freq="M", min_seg=8,
                         n_perm=2000, block=3, seed=42, verbose=True):
    """Data-driven break date per register, with a step-vs-drift check."""
    rng = np.random.default_rng(seed)
    if corpus is None:
        corpus = load_corpus()
    if groups is None:
        groups = list(TERM_GROUPS)

    def best_split(y):
        """Index and t statistic of the strongest two-sample split."""
        n = len(y)
        best_t, best_i = 0.0, None
        for i in range(min_seg, n - min_seg):
            a, b = y[:i], y[i:]
            va, vb = a.var(ddof=1), b.var(ddof=1)
            se = np.sqrt(va / len(a) + vb / len(b))
            if se == 0:
                continue
            t = abs(a.mean() - b.mean()) / se
            if t > best_t:
                best_t, best_i = t, i
        return best_i, best_t

    rows = []
    for g in groups:
        pats = TERM_GROUPS[g] if isinstance(g, str) else g
        s = _series_per10k(pats, corpus, freq, text_col, date_col)
        if len(s) < 2 * min_seg + 2:
            continue
        y = s.values.astype(float)
        i, t_obs = best_split(y)
        if i is None:
            continue

        # block permutation: shuffle blocks, keep local autocorrelation
        nb = int(np.ceil(len(y) / block))
        null = np.empty(n_perm)
        for k in range(n_perm):
            blocks = np.array_split(y, nb)
            order = rng.permutation(len(blocks))
            null[k] = best_split(np.concatenate([blocks[j] for j in order]))[1]
        p = (null >= t_obs).mean()

        # step vs drift
        x = np.arange(len(y), dtype=float)
        step = np.where(x < i, y[:i].mean(), y[i:].mean())
        slope, icept = np.polyfit(x, y, 1)
        line = icept + slope * x
        ss = ((y - y.mean()) ** 2).sum()
        r2_step = 1 - ((y - step) ** 2).sum() / ss
        r2_line = 1 - ((y - line) ** 2).sum() / ss
        abrupt = r2_step / r2_line if r2_line > 0 else np.inf

        rows.append({
            "group": g,
            "label": REGISTER_LABELS_TR.get(g, g),
            "date": s.index[i].strftime("%Y-%m"),
            "before": y[:i].mean(), "after": y[i:].mean(),
            "change_pct": (y[i:].mean() - y[:i].mean()) / y[:i].mean() * 100,
            "t": t_obs, "p": p, "abruptness": abrupt,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Benjamini-Hochberg across registers
    out = out.sort_values("p").reset_index(drop=True)
    m = len(out)
    out["q"] = np.minimum.accumulate(
        (out.p * m / (out.index + 1)).values[::-1])[::-1].clip(0, 1)
    out = out.sort_values("date")

    if verbose:
        print(f"{'kayıt':<28}{'kırılma':<10}{'önce':>7}{'sonra':>8}"
              f"{'değişim':>9}{'q':>8}   yapı")
        for r in out.itertuples():
            sig = "" if r.q < 0.05 else "  (anlamsız)"
            kind = ("basamak" if r.abruptness > 1.10 else
                    "kayma"   if r.abruptness < 0.95 else "belirsiz")
            print(f"  {r.label:<26}{r.date:<10}{r.before:>7.1f}{r.after:>8.1f}"
                  f"{r.change_pct:>8.0f}%{r.q:>8.3f}   {kind}{sig}")
    return out


# Olay adlarinin Turkce karsiligi. DISLOCATIONS anahtarlari kodda ingilizce
# kaliyor (analiz cikitisi), rapor tablolarinda bunlar kullaniliyor.
DISLOCATIONS_TR = {
    "Coup attempt 2016":     "Darbe girişimi, Tem. 2016",
    "Referendum 2017":       "Referandum, Nis. 2017",
    "General election 2018": "Genel seçim, Haz. 2018",
    "Local election 2019":   "Yerel seçim, Mar. 2019",
    "General election 2023": "Genel seçim, May. 2023",
    "Local election 2024":   "Yerel seçim, Mar. 2024",
    "Bahçeli's call 2024":   "Bahçeli çağrısı, Eki. 2024",
    "İmamoğlu arrest 2025":  "İmamoğlu tutuklaması, Mar. 2025",
    "Istanbul trustee 2025": "İstanbul kayyumu, Eyl. 2025",
}


def register_event_latex(group, corpus=None, table=None, caption=None,
                         label=None, drop_at=15.0, **kw):
    """LaTeX table of ONE register's behaviour around each event.

    Rows are events, so the table sits next to the finding it supports rather
    than in a single wide matrix at the end. `drop_at` is the per cent at which
    a move counts as a fall or a rise; below it the row reads as unchanged.

    Recovery is printed as `register_recovery` returns it, which means the
    censoring guard is carried through: for the last three events the corpus
    does not reach 24 months, and those cells say so instead of claiming the
    register never returned.
    """
    if table is None:
        table = register_recovery(group, corpus=corpus, verbose=False, **kw)
    name = REGISTER_LABELS_TR.get(group, group) if isinstance(group, str) else "Kayıt"
    caption = caption or f"{name}: olay çevresindeki hareket"
    label = label or ("tab:olay-" + (group if isinstance(group, str) else "kayit"))

    tr = {"no drop": "düşüş yok", "insufficient data": "veri yetersiz"}

    rows = []
    for r in table.itertuples():
        if pd.isna(r.change_pct):
            rows.append(f"  {DISLOCATIONS_TR.get(r.event, r.event)} & -- & -- & -- & \\duraga & -- \\\\")
            continue
        arrow = (r"\artan" if r.change_pct >= drop_at else
                 r"\azalan" if r.change_pct <= -drop_at else r"\duraga")
        rec = r.recovery
        if rec in tr:
            rec = tr[rec]
        elif rec.startswith("censored"):
            rec = "gözlem dışı"
        else:
            rec = rec.replace("months", "ay").replace("month", "ay")
        ad = DISLOCATIONS_TR.get(r.event, r.event)
        rows.append(f"  {ad} & {r.baseline:.1f} & {r.post:.1f} & "
                    f"{r.change_pct:+.0f}\\% & {arrow} & {rec} \\\\")

    return f"""\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
\\begin{{tabular}}{{@{{}}l r r r c l@{{}}}}
\\toprule
Olay & Taban & Sonrası & Değişim & & Dönüş \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\kaynaknot{{Taban olaydan önceki 12 ayın, \\emph{{Sonrası}} sonraki üç ayın
ortalamasıdır; ikisi de 10.000 kelimede vukuat sayısıdır. \\emph{{Dönüş}},
kaydın tabanın \\%90'ına döndüğü ay sayısıdır. Korpus 31 Mart 2026'da bittiği
için son üç olayda 24 aylık pencere dolmuyor: oralarda dönüş gözlem dışıdır,
gerçekleşmediği anlamına gelmez.}}
\\end{{table}}
"""


def register_change_venue_adjusted(groups=None, corpus=None, text_col="text_bert",
                                   date_col=DATE_COL, venue_col="venue",
                                   split_at=LEADER_CHANGE, labels=None,
                                   min_tokens=20000, verbose=True):
    """Register change with the venue mix held constant across both periods.

    The two periods are not made of the same kinds of speech. Özel gives more
    press statements and rallies, Kılıçdaroğlu more events and parliamentary
    speeches, so part of any difference between the periods could be a
    difference in where the speaking happened rather than in what was said.

    Post-stratification answers that. The rate is computed inside each venue,
    then the venues are recombined using one shared set of weights (the whole
    corpus mix). Both periods are then reported as if they had the same venue
    composition, and what remains cannot be venue.

    Venues holding fewer than `min_tokens` in either period are dropped and the
    weights renormalised, since a rate estimated on a handful of speeches would
    be noise carried by a large weight.
    """
    if corpus is None:
        corpus = load_corpus()
    if groups is None:
        groups = list(TERM_GROUPS)
    labels = labels or REGISTER_LABELS_TR

    df = corpus.dropna(subset=[date_col, text_col, venue_col]).copy()
    cut = pd.Timestamp(split_at)
    df["_per"] = np.where(df[date_col] < cut, "early", "late")

    # tokens and text per (period, venue)
    cell, toks = {}, {}
    for (per, ven), g in df.groupby(["_per", venue_col]):
        t = _tr_lower(" ".join(g[text_col].astype(str)))
        cell[(per, ven)] = t
        toks[(per, ven)] = len(_REG_TOK.findall(t))

    venues = sorted({v for _, v in cell})
    kept = [v for v in venues
            if toks.get(("early", v), 0) >= min_tokens
            and toks.get(("late", v), 0) >= min_tokens]
    dropped = [v for v in venues if v not in kept]

    # shared weights: each venue's share of the whole corpus
    tot = {v: toks.get(("early", v), 0) + toks.get(("late", v), 0) for v in kept}
    W = sum(tot.values())
    w = {v: tot[v] / W for v in kept}

    rows = []
    for g in groups:
        pats = TERM_GROUPS[g] if isinstance(g, str) else g
        raw, adj = {}, {}
        for per in ("early", "late"):
            n = sum(toks.get((per, v), 0) for v in kept)
            hits = sum(len(re.findall(p, cell.get((per, v), "")))
                       for v in kept for p in pats)
            raw[per] = hits / n * 10000 if n else np.nan
            adj[per] = sum(
                w[v] * sum(len(re.findall(p, cell[(per, v)])) for p in pats)
                / toks[(per, v)] * 10000 for v in kept)
        rows.append({
            "group": g, "label": labels.get(g, g),
            "raw_early": raw["early"], "raw_late": raw["late"],
            "raw_change": (raw["late"] - raw["early"]) / raw["early"] * 100,
            "adj_early": adj["early"], "adj_late": adj["late"],
            "adj_change": (adj["late"] - adj["early"]) / adj["early"] * 100,
        })

    out = pd.DataFrame(rows).sort_values("adj_change", ascending=False)
    out["shift"] = out.adj_change - out.raw_change

    if verbose:
        print(f"mekân sabitlendi. ağırlıklar: "
              + ", ".join(f"{v} {w[v]:.0%}" for v in kept))
        if dropped:
            print(f"dışarıda ({min_tokens:,} token altı): {', '.join(dropped)}")
        print(f"\n  {'kayıt':<26}{'ham':>9}{'düzelt.':>10}{'fark':>8}")
        for r in out.itertuples():
            # Orantisal esik: 1152 -> 1114 mutlak olarak 38 puan oynar ama
            # oransal olarak hicbir sey. Buyuk degisimlerde mutlak esik
            # yaniltici bayrak veriyor.
            rel = abs(r.shift) / max(abs(r.raw_change), 1)
            flag = "  <- mekân etkisi" if rel > 0.25 and abs(r.shift) > 8 else ""
            print(f"  {r.label:<26}{r.raw_change:>+8.0f}%{r.adj_change:>+9.0f}%"
                  f"{r.shift:>+8.0f}{flag}")
    return out
