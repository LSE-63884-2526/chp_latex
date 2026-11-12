"""
Contextual BERT embeddings for the main signifier terms.

Terim eslestirme bu dosyanin basindaki MATCHERS ile yapilir (default_stems'e ya
da sigdict'e BAGLI DEGIL, seffaf ve bagimsiz). Her hedef terim tek bir regex ile
tanimli; yorumlar hangi false positive'lerin dislandigini gosterir.

BLOCKERS: bir bolgeyi kilitler ama CIKTIYA GIRMEZ. "Cumhuriyet Halk Partisi"
icindeki "Halk"i halk sayilmaktan korur (en-uzun-oncelik: uzun blocker once
esleşir, bolgeyi kilitler, kisa halk oraya giremez).

Corpusa DOKUNULMAZ: regex sadece span'i etiketler, surface kolonu ORIJINAL formu
tutar (milletin, hakkini, turkiyenin). En-uzun-oncelik + occ kilidi ust uste
eslesmeyi onler.

15 ve 20 context window AYRI dosyalara uretilir:
    {term}_cw15tvectors.npy / {term}_cw15metadata.csv
    {term}_cw20tvectors.npy / {term}_cw20metadata.csv
"""

import os, re
import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================
# BLOCKERS  (bolgeyi kilitler, cikti vermez -> halk'i CHP'den korur)
# =============================================================================
BLOCKERS = [
    r"cumhuriyet halk parti\w*",
    r"\bchp\b",
    r"adalet ve kalkınma parti\w*",   # AKP tam adi -> adalet sayma
    r"\bak\s*parti\w*",               # AK Parti
    r"\btbmm\b",
    r"türkiye büyük millet meclisi\w*",
    r"büyük millet meclisi\w*",
    r"millet meclisi\w*",
    r"birleşmiş milletler\w*",
    r"\bbm\b",
    r"millet ittifakı\w*",
    r"milletin meclisi\w*",
    r"milletin vekili\w*",
    r"bilal erdoğan\w*",
    r"emine erdoğan\w*",
    r"sümeyye erdoğan\w*",
]

# =============================================================================
# HEDEF TERIMLER  (tek kaynak: burada tanimli, disariya bagli degil)
# Anahtar = base etiket (dosya adi + metadata'daki term). Deger = regex.
# Regex tr_lower'lanmis metne uygulanir (kucuk harf, Turkce i/I).
# =============================================================================
ET_RE = re.compile(r"^e[td]")

# YENI: STYLE TOKENS (mask_style=True iken attention'dan ve ortalamadan dislanir)
# Soru ekleri regex degil SET: "mi\w*" yazilsaydi "millet"i de yakalardi.
Q_WORDS = {"niye", "neden", "niçin", "nere", "nereye", "nasıl", "ne", "neye", "neyi",
           "mı", "mi", "mu", "mü", "mısın", "misin", "musun", "müsün",
           "mıyız", "miyiz", "muyuz", "müyüz", "mısınız", "misiniz", "musunuz", "müsünüz"}
SAY_RE = re.compile(r"^(hangi|diyor|diyen|vallah|billah|dedi(?!kodu)|dediler|demiş|söyl(?!em(?:ler\w*|i|in|ini|de|den|le)?$))\w*")
SAY_EXACT = {"der", "derler"}
VOCATIVE = {"sayın", "sevgili", "değerli", "kardeşim", "kardeşlerim",
            "efendim", "arkadaşlar", "arkadaşlarım", "kardeşlerim", "kıymetli"}
GREETINGS = {"merhaba", "selam", "selamlar", "selamlıyorum", "selamlarımla",
             "teşekkür", "teşekkürler","saygılar", "saygılarımla", "sunuyorum"}
KK_STOP = {"bakın", "lazım"}

def is_style(w):
    w = w.strip('.!?,;:"“”‘’\'()[]')
    return w in Q_WORDS or w in SAY_EXACT or w in VOCATIVE or w in GREETINGS or w in KK_STOP or bool(SAY_RE.match(w))

MATCHERS = {
    # millet (halk/ulus). milletvekili ve milletlerarasi HARIC (negatif lookahead).
    "millet":     r"\bmillet(?!vekil|lerarası)\w*",

    # hak (right/justice). Sik gecen false positive'ler HARIC:
    #   hakkari (sehir), hakem, hâkim/hakim (yargic), hakkında/hakkımda/hakkımızda/
    #   hakkınızda (about), haklarında (about them), hakikat (truth), hakaret (insult),
    #   haksiz (unfair), hakan (Hakan Fidan).
    #"hak":        r"\bhak(?!kari|ir|em|kâri|im|kında|kımda|kımızda|kınızda|larında|ikat|aret|sız|an|lı|kaniyet|[-\s]?iş)\w*",

    # halk (people). CHP parti adi BLOCKERS ile korunuyor (yukari bak).
    #"halk":       r"\bhalk\w*",

    # demokrasi. demokra* koku: demokrasi + demokrat + demokratik hepsi dahil.
    #"demokrasi":  r"\bdemokra\w*",

    # türkiye (ulke). türki* : türkiye ailesi (türkiyenin, türkiyeye...). "türk"
    # (etnik/sifat, 4 harf) yakalanmaz cunku pattern "türki" (5 harf) ister.
    #"türkiye":    r"\btürki\w*",

    # erdoğan. Çok kaynaklı: tam ad, tayyip, recep, tayyip bey hepsi erdoğan.
    "erdoğan":    r"\b(?:recep\s+tayyip\s+erdoğan|tayyip\s+erdoğan|recep\s+tayyip|tayyip\s+bey|erdoğan|tayyip)\w*",

    #"çocu[kğ]": r"\bçocuk(?!luk)\w*",
    #"adalet":   r"\badalet(?!siz)\w*",
    #"emekli":   r"\bemekli(?!lik)\w*",
    #"kadın":    r"\bkadın(?!lık)\w*",
    #"işçi":     r"\bişçi(?!lik|sizlik)\w*",
    #"aile":     r"\baile(?!vi)\w*",
    #"genç":     r"\bgenç(?!lik)\w*",
    #"çiftçi":   r"\bçiftçi(?!lik)\w*",


    # ÇALIŞTIRMADAN ÖNCE PATH DEĞİŞTİR: TALEPLER
    # "lira": r"\b"
    #"sorun": 
    # "para":

    # ÇALIŞTIRMADAN ÖNCE PATH DEĞİŞTİR: BELEDİYE
    #"istanbul":
    # buraya ekrem imamoğlu da eklemek istiyorum
}

# ----------------- CONFIG (Windows yollarini kendine gore degistir) -----------
BERT_CSV   = r"C:\Users\user\Desktop\Thesis\Data\corpus_main_processed.csv"  
OUT_DIR    = r"C:\Users\user\Desktop\Thesis\Embeddings"
MODEL_NAME = "dbmdz/bert-base-turkish-uncased"

TEXT_COL   = "text_bert"
WINDOWS    = [15]     # iki pencere: ayri dosyalara
BATCH_SIZE = 64
MAX_LEN    = 160
MIN_OCC    = 20
# -------------------------------------------------------------------------------
WHITEN_K = 256   # whitening'de tutulan bilesen sayisi (gozlem sayisindan kucuk olmali)

def whiten(X, k=WHITEN_K, eps=1e-6):
    # DEGISIKLIK 3: sadece en guclu k bilesen tutulur.
    # Sifira yakin ozdegerler 1/sqrt(eps) ile buyutulmez, gurultu patlamaz.
    k = min(k, X.shape[0] - 1)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1][:k]
    W = eigvecs[:, idx] / np.sqrt(eigvals[idx] + eps)
    return (Xc @ W).astype(np.float32)

def tr_lower(t):
    return (str(t).replace("İ","i").replace("I","ı").replace("Ğ","ğ")
            .replace("Ş","ş").replace("Ç","ç").replace("Ö","ö").replace("Ü","ü").lower())

def phase_of(ts):
    if pd.isna(ts): return 0
    if ts < pd.Timestamp("2019-04-01"): return 1   # KK erken
    if ts < pd.Timestamp("2023-11-05"): return 2   # KK gec
    return 3                                        # Ozel

# ---- MATCHERS / BLOCKERS derle ----
def build_matcher():
    """base -> compiled regex. Tek kaynak MATCHERS."""
    return {base: re.compile(pat) for base, pat in MATCHERS.items()}

def build_blockers():
    """compiled blocker regex listesi. Tek kaynak BLOCKERS."""
    return [re.compile(pat) for pat in BLOCKERS]

WORD_RE = re.compile(r"\S+")

def doc_words(text):
    """metni bir kez tara: kelimeler + (start,end) ofsetleri."""
    words, starts, ends = [], [], []
    for m in WORD_RE.finditer(text):
        words.append(m.group()); starts.append(m.start()); ends.append(m.end())
    return words, starts, ends

def span_to_words(starts, ends, s, e):
    """char araligi [s,e) ile ortusen kelime index'leri (bisect ile hizli)."""
    import bisect
    lo = bisect.bisect_right(ends, s)
    out, i = [], lo
    while i < len(starts) and starts[i] < e:
        if ends[i] > s:
            out.append(i)
        i += 1
    return out

def find_occurrences(tl, n, matchers, blockers):
    """tl: tr_lower(text). Doner: [(char_start, char_end, term), ...]
    Blocker'lar bolgeyi kilitler ama CIKTIYA GIRMEZ (base=None).
    En uzun eslesme once alinir, o bolge kilitlenir (occ), ust uste eslesme olmaz.
    Uzun blocker ("cumhuriyet halk parti") kisa matcher'dan ("halk") once
    kilitledigi icin halk CHP bolgesine giremez."""
    matches = []
    # blocker'lar: base=None -> kilitle, cikti verme
    for p in blockers:
        for m in p.finditer(tl):
            matches.append((m.start(), m.end(), None))
    # matcher'lar: base=terim adi
    for base, p in matchers.items():
        for m in p.finditer(tl):
            matches.append((m.start(), m.end(), base))
    matches.sort(key=lambda x: (-(x[1] - x[0]), x[0]))   # en uzun oncelikli
    occ = bytearray(n)
    out = []
    for s, e, base in matches:
        if any(occ[s:e]):
            continue
        for k in range(s, e):
            occ[k] = 1
        if base is not None:                 # blocker degilse ciktiya ekle
            out.append((s, e, base))
    out.sort(key=lambda x: x[0])
    return out

# Ayni signifier'in, hedefleri < MERGE_WINDOW kelime yakininda olan occurrence'lari
# TEK gozleme birlesir (anchor kurali; uzak occurrence'lar zincirlenmez). Boylece
# tek bir retorik pasaj ("millet, millet, millet") birden fazla satir sayilmaz.
MERGE_WINDOW = 30

def build_jobs(df, window, matchers, blockers, merge=True, W_merge=MERGE_WINDOW):
    jobs = []
    seen = {t: 0 for t in matchers}     # birlesme SONRASI gozlem sayisi
    raw  = {t: 0 for t in matchers}     # birlesme ONCESI ham occurrence sayisi (log icin)
    for _, row in df.iterrows():
        text = row.get(TEXT_COL)
        if not isinstance(text, str) or not text.strip():
            continue
        tl = tr_lower(text)
        words, starts, ends = doc_words(text)

        meta_base = {"doc_id": row.get("id"), "date": row.get("date"),      # BURADA
                     "year": row.get("year"), "month": row.get("month"),
                     "venue": row.get("venue"), "phase": row["_phase"]}

        # 1) bu dokumandaki occurrence'lari term bazinda topla (kelime-index span'leri)
        occ_by_term = defaultdict(list)
        for s, e, term in find_occurrences(tl, len(tl), matchers, blockers):
            tgt = span_to_words(starts, ends, s, e)
            if tgt:
                occ_by_term[term].append(tgt)
                raw[term] += 1

        # 2) term bazinda anchor-grupla, her grup icin TEK job uret
        for term, occ_list in occ_by_term.items():
            occ_list.sort(key=lambda t: t[0])
            if merge:
                groups, cur, anchor = [], [], None
                for tgt in occ_list:
                    p = tgt[0]
                    if anchor is None or (p - anchor) < W_merge:
                        if anchor is None:
                            anchor = p
                        cur.append(tgt)
                    else:
                        groups.append(cur); cur = [tgt]; anchor = p
                if cur:
                    groups.append(cur)
            else:
                groups = [[tgt] for tgt in occ_list]     # merge kapali -> eski davranis

            for g in groups:
                all_idx = [i for tgt in g for i in tgt]
                lo = max(0, min(all_idx) - window)
                hi = min(len(words), max(all_idx) + window + 1)
                ctx = list(words[lo:hi])
                tgt_groups = []                          # her mention icin lokal index listesi
                for tgt in g:
                    loc = [i - lo for i in tgt]
                    for j in loc:
                        ctx[j] = ctx[j].strip('.!?,;:"“”‘’\'()[]')
                    tgt_groups.append(loc)
                surface = " | ".join(" ".join(ctx[j] for j in loc) for loc in tgt_groups)
                meta = dict(meta_base, surface=surface, context=" ".join(ctx),
                            n_merged=len(g))
                jobs.append((ctx, tgt_groups, term, meta))
                seen[term] += 1

    return jobs, seen, raw

# ----------------- BURADAN ASAGISI MODELI GEREKTIRIR -----------------
def run(test_only=False, test_df=None, mask_target=False, mask_et=False, ctx_embed=False,
        mask_style=False):
    # ctx_embed=True: DEGISIKLIK 4. Hedef kelime yerine BAGLAMIN vektoru alinir
    # (hedef + sagindaki/solundaki 1 kelime haric). Tematik kumeler icin.
    # mask_style=True: YENI. Soru kaliplari ve diyor/soyluyor attention'dan ve
    # ctx_embed ortalamasindan dislanir, "?" -> "." yapilir.

    if ctx_embed:
        mode = "CONTEXT EMBEDDING (target +-1 excluded, no whitening)"
    elif mask_target:
        mode = "MASKED TARGET + WHITENED"
    elif mask_et:
        mode = "MASKED HAK ET (DESERVE) + WHITENED"
    else:
        mode = "NORMAL (unmasked, no whitening)"
    if mask_style:
        mode += " + STYLE TOKENS IGNORED"
    print(f"\n{'='*60}\n  MODE: {mode}\n{'='*60}")
    
    df = test_df if test_df is not None else pd.read_csv(BERT_CSV)
    df["_dt"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["_phase"] = df["_dt"].apply(phase_of)

    matchers = build_matcher()
    blockers = build_blockers()

    if test_only:
        # model olmadan: ilk pencereyle is listesi + sayim
        return build_jobs(df, WINDOWS[0], matchers, blockers)

    # --- model bir kez yuklenir, iki pencere icin de kullanilir ---
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    os.makedirs(OUT_DIR, exist_ok=True)

    from collections import defaultdict

    MASK_ID = tok.mask_token_id   # [MASK] token'inin id'si

    
    @torch.no_grad()
    def flush(batch, vecs, metas, mask_target=False, mask_et=False, ctx_embed=False,
              mask_style=False):
        # DEGISIKLIK 2: tokenizer'a Turkce kucuk harfli kelimeler gider (I -> ı, İ -> i)
        # YENI: mask_style=True ise "?" -> "." (cumle yapisi bozulmaz, soru sinyali gider)
        ctxs = [[tr_lower(w).replace("?", ".") if mask_style else tr_lower(w) for w in b[0]]
                for b in batch]
        enc = tok(ctxs, is_split_into_words=True, padding=True,
                  truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    
        if mask_target or mask_et:
            for bi, (ctx, tgt_groups, term, meta) in enumerate(batch):
                wids = enc.word_ids(bi)
                target_word_ids = set(i for loc in tgt_groups for i in loc)
    
                # "hak et-" kirma: hedefin hemen sagindaki kelime et- fiili ise onu da maskele
                et_word_ids = set()
                if mask_et and term == "hak":
                    for wi in target_word_ids:
                        nxt = wi + 1
                        if nxt < len(ctx) and ET_RE.match(tr_lower(ctx[nxt])):
                            et_word_ids.add(nxt)
    
                for k, w in enumerate(wids):
                    if mask_target and w in target_word_ids:
                        enc["input_ids"][bi, k] = MASK_ID
                    if mask_et and w in et_word_ids:
                        enc["input_ids"][bi, k] = MASK_ID

        # YENI: style token'lari attention_mask=0 -> diger tokenlar onlara bakmaz,
        # pozisyonlar ve cumle uzunlugu degismez. style_ids ctx_embed ortalamasinda da kullanilir.
        style_ids = [set() for _ in batch]
        if mask_style:
            for bi, ctx in enumerate(ctxs):
                style_ids[bi] = {i for i, w in enumerate(ctx) if is_style(w)}
                for k, w in enumerate(enc.word_ids(bi)):
                    if w in style_ids[bi]:
                        enc["attention_mask"][bi, k] = 0

        # son 4 katmanin ortalamasi: son katmana gore yuzey bicime daha az bagli
        out = model(**enc, output_hidden_states=True)
        hid = torch.stack(out.hidden_states[-4:]).mean(0)
        for bi, (ctx, tgt_groups, term, meta) in enumerate(batch):
            wids = enc.word_ids(bi)

            if ctx_embed:
                # DEGISIKLIK 4: hedef kelime ve iki yanindaki birer kelime haric,
                # kalan baglam tokenlarinin ortalamasi
                # YENI: style token'lari da ortalamaya girmez
                excl = {j + d for loc in tgt_groups for j in loc for d in (-1, 0, 1)} | style_ids[bi]
                ctx_pos = [k for k, w in enumerate(wids) if w is not None and w not in excl]
                if not ctx_pos:
                    continue
                v = hid[bi, ctx_pos].mean(0).cpu().numpy().astype(np.float32)
                vecs[term].append(v); metas[term].append(meta)
                continue

            occ_vecs = []
            for loc in tgt_groups:
                # DEGISIKLIK 1: sadece ilk alt parca (kok). ##dır, ##lar gibi ekler alinmaz
                pos = [k for k, w in enumerate(wids) if w in loc][:1]
                if not pos:
                    continue
                occ_vecs.append(hid[bi, pos].mean(0))
            if not occ_vecs:
                continue
            v = torch.stack(occ_vecs).mean(0).cpu().numpy().astype(np.float32)
            vecs[term].append(v); metas[term].append(meta)

    # --- her pencere icin ayri kosum, ayri dosya ---
    for window in WINDOWS:
        print(f"\n=== context window {window} ===")
        jobs, seen, raw = build_jobs(df, window, matchers, blockers)
        print(f"  Total {len(jobs)} merged observations "
              f"(from {sum(raw.values())} raw occurrences):")
        for term, n in seen.items():
            print(f"    {term:12} {raw[term]:6} -> {n:6}")

        vecs = defaultdict(list); metas = defaultdict(list)
        buf = []
        for job in jobs:
            buf.append(job)
            if len(buf) == BATCH_SIZE:
                flush(buf, vecs, metas, mask_target=mask_target, mask_et=mask_et, ctx_embed=ctx_embed,
                      mask_style=mask_style); buf = []
        if buf:
            flush(buf, vecs, metas, mask_target=mask_target, mask_et=mask_et, ctx_embed=ctx_embed,
                  mask_style=mask_style)

        if ctx_embed:
            tag = "_ctx"
        elif mask_target:
            tag = "_maskwhite"
        elif mask_et:
            tag = "_masket"
        else:
            tag = ""
        # YENI: eski dosyalarin ustune yazmasin
        if mask_style:
            tag += "_nostyle"
        saved = 0
        
        for term in vecs:
            if len(vecs[term]) < MIN_OCC:
                continue
            do_whiten = (mask_target or mask_et) and not ctx_embed
            arr = np.vstack(vecs[term])
            if do_whiten:
                off_before = np.abs(np.cov(arr, rowvar=False) - np.diag(np.diag(np.cov(arr, rowvar=False)))).mean()
                arr = whiten(arr)
                off_after = np.abs(np.cov(arr, rowvar=False) - np.diag(np.diag(np.cov(arr, rowvar=False)))).mean()
                print(f"    {term}: whiten | off-diagonal {off_before:.4f} -> {off_after:.4f}")
            safe = term.replace("/", "_")
            np.save(os.path.join(OUT_DIR, f"{safe}_cw{window}vectors{tag}.npy"), arr)
            pd.DataFrame(metas[term]).to_csv(
                os.path.join(OUT_DIR, f"{safe}_cw{window}metadata{tag}.csv"), index=False)
            saved += 1

        zero = sorted(t for t in seen if seen[t] == 0)
        if zero:
            print(f"  UYARI cw{window} 0 occurrence:", zero)
        print(f"  cw{window}: {saved} terms")


    print(f"\nFinished. Documents -> {OUT_DIR}")

#run(mask_target=False)
#run(mask_target=False)
#run(mask_et=False)
#run(ctx_embed=True)   # baglam vektorleri (style dahil)
run(ctx_embed=True, mask_style=True)   # YENI: baglam vektorleri, style token'lari haric