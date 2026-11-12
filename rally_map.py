import re
import pandas as pd
import folium

# ---- AYARLAR ----
CSV = "/Users/erenisikavci/Desktop/Thesis/Data/corpus_main.csv"
TEXT_COL = "text"        
VENUE_COL = "venue"
N_WORDS = 500                  # ilk kac kelimeye bakilacak
MIN_HITS = 1                   # sehir en az kac kez gecmeli (gurultu filtresi)
VENUE_KEEP = ["rally"]

# ---- 81 IL KOORDINAT ----
PROV = {
"adana":(37.00,35.32),"adiyaman":(37.76,38.28),"afyonkarahisar":(38.76,30.54),
"agri":(39.72,43.05),"amasya":(40.65,35.83),"ankara":(39.93,32.85),
"antalya":(36.90,30.70),"artvin":(41.18,41.82),"aydin":(37.85,27.85),
"balikesir":(39.65,27.88),"bilecik":(40.14,29.98),"bingol":(38.88,40.50),
"bitlis":(38.40,42.11),"bolu":(40.74,31.61),"burdur":(37.72,30.29),
"bursa":(40.19,29.06),"canakkale":(40.15,26.41),"cankiri":(40.60,33.62),
"corum":(40.55,34.95),"denizli":(37.78,29.09),"diyarbakir":(37.91,40.24),
"edirne":(41.68,26.56),"elazig":(38.68,39.22),"erzincan":(39.75,39.49),
"erzurum":(39.90,41.27),"eskisehir":(39.78,30.52),"gaziantep":(37.07,37.38),
"giresun":(40.91,38.39),"gumushane":(40.46,39.48),"hakkari":(37.57,43.74),
"hatay":(36.20,36.16),"isparta":(37.76,30.55),"mersin":(36.81,34.64),
"istanbul":(41.01,28.98),"izmir":(38.42,27.14),"kars":(40.60,43.09),
"kastamonu":(41.38,33.78),"kayseri":(38.73,35.49),"kirklareli":(41.74,27.22),
"kirsehir":(39.15,34.16),"kocaeli":(40.85,29.88),"konya":(37.87,32.48),
"kutahya":(39.42,29.98),"malatya":(38.35,38.31),"manisa":(38.61,27.43),
"kahramanmaras":(37.58,36.93),"mardin":(37.31,40.74),"mugla":(37.22,28.36),
"mus":(38.73,41.49),"nevsehir":(38.62,34.71),"nigde":(37.97,34.68),
"ordu":(40.98,37.88),"rize":(41.02,40.52),"sakarya":(40.77,30.40),
"samsun":(41.29,36.33),"siirt":(37.93,41.94),"sinop":(42.03,35.15),
"sivas":(39.75,37.02),"tekirdag":(40.98,27.51),"tokat":(40.31,36.55),
"trabzon":(41.00,39.72),"tunceli":(39.11,39.55),"sanliurfa":(37.17,38.79),
"usak":(38.68,29.41),"van":(38.49,43.38),"yozgat":(39.82,34.81),
"zonguldak":(41.45,31.79),"aksaray":(38.37,34.03),"bayburt":(40.26,40.22),
"karaman":(37.18,33.22),"kirikkale":(39.85,33.51),"batman":(37.88,41.13),
"sirnak":(37.52,42.46),"bartin":(41.63,32.34),"ardahan":(41.11,42.70),
"igdir":(39.92,44.04),"yalova":(40.66,29.28),"karabuk":(41.20,32.62),
"kilis":(36.72,37.12),"osmaniye":(37.07,36.25),"duzce":(40.84,31.16),
}

# alt isimler -> resmi il
ALIAS = {
"afyon":"afyonkarahisar","urfa":"sanliurfa","maras":"kahramanmaras",
"antep":"gaziantep","izmit":"kocaeli","dersim":"tunceli","adapazari":"sakarya",
"icel":"mersin","stimbul":"istanbul", "marmaris":"mugla","bodrum":"mugla","fethiye":"mugla","datca":"mugla",
"boyabat":"sinop","mudanya":"bursa","selcuk":"izmir","odemis":"izmir", "elmadağ": "ankara",
"sivrihisar":"eskisehir","polatli":"ankara","golbasi":"ankara","sincan":"ankara",
"didim":"aydin","soke":"aydin","nazilli":"aydin","kusadasi":"aydin",
"unye":"ordu","fatsa":"ordu","alanya":"antalya","corlu":"tekirdag",
"luleburgaz":"kirklareli","babaeski":"kirklareli","akhisar":"manisa",
"bakirkoy":"istanbul","atasehir":"istanbul","besiktas":"istanbul",
}

def norm(s):
    return str(s).replace("İ","i").replace("I","ı").lower()

# turkce -> ascii (koordinat anahtarlariyla eslesmek icin)
def deacc(s):
    tbl = str.maketrans("çğıöşü","cgiosu")
    return norm(s).translate(tbl)

PROV_KEYS = set(PROV.keys())

AMBIGUOUS = {"ordu","aydin","usak","van","kars","mus"}

NAME_BLOCK = {"aydin": ["koray"]}  # koray aydin = isim, il degil

def extract_city(text):
    toks = re.findall(r"[a-zçğıöşü]+", norm(text))[:N_WORDS]
    toks = [deacc(t) for t in toks]
    counts = {}
    for i, t in enumerate(toks):
        key = ALIAS.get(t, t)
        if key in PROV_KEYS:
            # onceki token isim bloklamada mi
            if key in NAME_BLOCK and i > 0 and toks[i-1] in NAME_BLOCK[key]:
                continue
            counts[key] = counts.get(key, 0) + 1
    counts = {c: h for c, h in counts.items() if not (c in AMBIGUOUS and h < 2)}
    if not counts:
        return None
    return max(counts.items(), key=lambda x: x[1])[0]
    return city

# ---- CALISTIR ----
df = pd.read_csv(CSV)
print("venue degerleri:")
print(df[VENUE_COL].value_counts())
print("---")

df = df[df[VENUE_COL].isin(VENUE_KEEP)]

df["city"] = df[TEXT_COL].apply(extract_city)

hit = df["city"].notna().mean()
print(f"sehir bulunan oran: {hit:.1%}")
print("---")
counts = df["city"].value_counts()
print(counts)

# ---- HARITA ----
m = folium.Map(location=[39.0,35.0], zoom_start=6, tiles="cartodbpositron")
mx = counts.max()
for city, n in counts.items():
    lat, lon = PROV[city]
    folium.CircleMarker(
        location=[lat,lon],
        radius=4 + 20*(n/mx),
        popup=f"{city}: {n}",
        color="#c0392b", fill=True, fill_opacity=0.6,
    ).add_to(m)


#####


import re
import unicodedata
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# df: city + speaker kolonlu dataframe bellekte acik olmali
GEOJSON = "/Users/erenisikavci/Desktop/Thesis/turkiye_iller.geojson"
OUT = "/Users/erenisikavci/Desktop/chp_report_current/chp_latex/figures/miting.png"

SPEAKER_COL = "speaker"

def ascii_key(s):
    s = str(s).replace("İ","i").replace("I","ı").lower()
    s = s.translate(str.maketrans("çğıöşü","cgiosu"))
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()

GEO_FIX = {"afyon":"afyonkarahisar","maras":"kahramanmaras","urfa":"sanliurfa","icel":"mersin"}

g = gpd.read_file(GEOJSON)
g["key"] = g["name"].apply(ascii_key).apply(lambda k: GEO_FIX.get(k, k))

# only demonstrations
dm = df[df["venue"] == "rally"]

# iki alt kume
kk = dm[dm[SPEAKER_COL].str.contains("kilicdaroglu", case=False, na=False)]
oz = dm[dm[SPEAKER_COL].str.contains("özel|ozel", case=False, na=False)]

def counts_map(sub):
    c = sub["city"].value_counts()
    return {ascii_key(k): v for k, v in c.items()}

cnt_kk = counts_map(kk)
cnt_oz = counts_map(oz)

g["kk"] = g["key"].map(cnt_kk).fillna(0).astype(int)
g["oz"] = g["key"].map(cnt_oz).fillna(0).astype(int)

# ORTAK skala: iki panelin max'i
vmax = max(g["kk"].max(), g["oz"].max())
norm = PowerNorm(gamma=0.4, vmin=1, vmax=vmax)                 # 0.55 -> 0.4, dusuk iller yukari
cmap = LinearSegmentedColormap.from_list("chp", ["#fcc9c0", "#e74c3c", "#7b1010"])  # beyaz -> acik pembe baslangic

ZERO_COLOR = "#e8e8e8"   # 0 iller belirgin gri, pembe 1'den ayrilsin

# donem sureleri (ay)
KK_MONTHS = 106   # 2015-01 -> 2023-11, kabaca
OZ_MONTHS = 32    # 2023-11 -> 2026-07, kabaca

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

panels = [
    (axes[0], "kk",  f"Kılıçdaroğlu (2015-2023, ~{KK_MONTHS} ay)\nn={int(g['kk'].sum())}"),
    (axes[1], "oz",  f"Özel (2023-2026, ~{OZ_MONTHS} ay)\nn={int(g['oz'].sum())}"),
]

for ax, col, title in panels:
    g.plot(column=col, cmap=cmap, norm=norm, edgecolor="#999999", linewidth=0.40, ax=ax, legend=False)
    g[g[col]==0].plot(color=ZERO_COLOR, edgecolor="#999999", linewidth=0.25, ax=ax)   # 0 iller gri
    # her panelde kendi top 8 etiketi
    top = g[g[col]>0].nlargest(15, col)
    for _, r in top.iterrows():
        c = r.geometry.centroid
        ax.annotate(r["name"], xy=(c.x, c.y), ha="center", va="center",
                    fontsize=6, fontweight="bold", color="black")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=6)

# ortak colorbar alta
sm = ScalarMappable(cmap=cmap, norm=norm)
# colorbar icin elle konum: [sol, alt, genislik, yukseklik]
cax = fig.add_axes([0.35, 0.20, 0.30, 0.025])
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Miting sayısı", fontsize=9)
cbar.set_ticks([1, 15, 30, 45, 60])
cbar.ax.tick_params(labelsize=8)

plt.subplots_adjust(top=0.92, bottom=0.16, wspace=0.02)
plt.savefig(OUT, dpi=600, bbox_inches="tight")
print("kaydedildi:", OUT)
print("KK n:", g["kk"].sum(), "| Özel n:", g["oz"].sum())



