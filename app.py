"""
SYD Score Backend — Scoring-based Hoax Yield Detection
Fuzzy Logic Mamdani Engine + Weighted Scoring
Author: SYD Score Team
"""

from flask import Flask, request, jsonify
import re
import math
import traceback

try:
    from flask_cors import CORS
    _has_cors = True
except ImportError:
    _has_cors = False

app = Flask(__name__)

if _has_cors:
    CORS(app)
else:
    # Manual CORS header injection tanpa flask-cors
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response

    @app.route('/analisis', methods=['OPTIONS'])
    def options_analisis():
        from flask import Response
        return Response(status=200)

# ─────────────────────────────────────────────
# LEKSIKON — Kamus Kata untuk Analisis Linguistik
# Disusun berdasarkan penelitian Putri et al. (2025), 
# Jurnal Predikat Adzkia (2024), dan Dewan Pers Indonesia
# ─────────────────────────────────────────────

KATA_EMOSIONAL = [
    # Rasa takut / kepanikan
    "gawat", "bahaya", "darurat", "panik", "mengerikan", "menyeramkan", "ngeri",
    "menakutkan", "seram", "horor", "kiamat", "bencana", "malapetaka", "krisis",
    "ancaman", "teror", "waspada", "siaga", "awas", "hati-hati",
    # Kemarahan
    "marah", "murka", "berang", "geram", "benci", "benci", "laknat", "keparat",
    "bajingan", "bangsat", "celaka", "sial", "jahat", "busuk", "korup",
    # Kesedihan / dramatisasi
    "sedih", "hancur", "musnah", "binasa", "lenyap", "hilang", "tragis",
    "memilukan", "menyayat", "miris", "menyedihkan", "pilu",
    # Kekhawatiran
    "mengkhawatirkan", "mengancam", "meresahkan", "meresahkan", "mengganggu",
    "memprihatinkan", "memburuk", "parah", "kritis", "genting",
    # Rasa bersalah / tuduhan
    "salah", "dosa", "berdosa", "bersalah", "menipu", "bohong", "palsu",
    "licik", "curang", "serakah", "rakus", "zalim", "aniaya",
]

KATA_HASUTAN = [
    # Ajakan menyebarkan
    "sebarkan", "bagikan", "share", "forward", "kirimkan", "sebar",
    "teruskan", "sampaikan kepada", "beritahukan", "beri tahu semua",
    "viralkan", "sebarluaskan", "sampaikan ke semua",
    # Ajakan bertindak dengan urgensi negatif
    "jangan percaya", "jangan mau", "tolak", "lawan", "boikot",
    "hentikan", "stop", "larang", "cegah", "blokir",
    # Provokasi kelompok
    "kafir", "sesat", "murtad", "anti islam", "anti agama", "anti nasional",
    "komunis", "pki", "antek", "kaki tangan", "boneka",
    # Konspirasi
    "konspirasi", "diam-diam", "secara rahasia", "disembunyikan",
    "ditutup-tutupi", "disembunyikan oleh", "tidak diberitahu",
    "dari dalam istana", "dari dalam lingkaran",
    # Sensasi berbahaya
    "sebelum terlambat", "sebelum dihapus", "sebelum diblokir",
    "sebelum hilang", "sebelum ketahuan", "segera sebelum",
]

KATA_HIPERBOLA = [
    # Kata bombastis
    "luar biasa", "fantastis", "menakjubkan", "spektakuler", "sensasional",
    "revolusioner", "monumental", "fenomenal", "legendaris",
    # Superlatif ekstrem
    "paling", "terhebat", "terbesar", "terkuat", "tertinggi", "terendah",
    "paling hebat", "paling canggih", "paling berbahaya", "terbukti",
    "100 persen", "dijamin", "pasti", "mutlak", "absolut",
    # Hiperbolik
    "hancur lebur", "porak poranda", "luluh lantak", "habis-habisan",
    "total hancur", "musnah sepenuhnya", "punah", "meledak",
    "banjir darah", "lautan air mata",
    # Klaim palsu / tidak berdasar
    "sudah terbukti", "terbukti secara ilmiah", "dokter menemukan",
    "para ilmuwan kaget", "mengejutkan dunia", "mengubah segalanya",
    "orang kaya tidak mau", "mereka tidak mau kamu tahu",
    # Urgensi palsu
    "sekarang juga", "detik ini", "jam ini", "hari ini juga",
    "langsung", "segera", "jangan tunda", "tidak boleh terlambat",
]

# Media resmi Indonesia yang terdaftar Dewan Pers
MEDIA_RESMI_INDONESIA = {
    "kompas.com", "kompas.id", "detik.com", "tribunnews.com", "liputan6.com",
    "tempo.co", "republika.co.id", "cnnindonesia.com", "cnbcindonesia.com",
    "antara.net", "antaranews.com", "merdeka.com", "okezone.com",
    "sindonews.com", "beritasatu.com", "mediaindonesia.com", "suarasurabaya.net",
    "bisnis.com", "katadata.co.id", "tirto.id", "thejakartapost.com",
    "jpnn.com", "suara.com", "kumparan.com", "grid.id", "iNews.id",
    "inews.id", "viva.co.id", "rmol.id", "rakyatmerdeka.co.id",
    "harianmerdeka.id", "pojoksatu.id", "wartaekonomi.co.id",
    # Lembaga pemerintah & resmi
    "go.id", "ac.id", "sch.id", "mil.id", "net.id",
    "who.int", "un.org", "reuters.com", "apnews.com", "bbc.com",
    "aljazeera.com", "bloomberg.com", "ft.com",
}

# ─────────────────────────────────────────────
# TRIANGULAR MEMBERSHIP FUNCTION
# mu(x; a,b,c) — fungsi keanggotaan segitiga
# ─────────────────────────────────────────────

def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular Membership Function.
    Parameter: a (kaki kiri), b (puncak), c (kaki kanan)
    Return: derajat keanggotaan [0, 1]
    """
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        if b == a:
            return 1.0
        return (x - a) / (b - a)
    else:  # b < x < c
        if c == b:
            return 1.0
        return (c - x) / (c - b)

def fuzzify(value: float) -> dict:
    """
    Fuzzifikasi nilai input ke tiga himpunan linguistik:
    RENDAH  (a=0,  b=0,  c=50)
    SEDANG  (a=25, b=50, c=75)
    TINGGI  (a=50, b=100, c=100)
    """
    # Fungsi RENDAH: trapezoid kiri (0→0→50)
    if value <= 0:
        mu_rendah = 1.0
    elif value <= 25:
        mu_rendah = 1.0
    elif value <= 50:
        mu_rendah = (50 - value) / 25.0
    else:
        mu_rendah = 0.0

    # Fungsi SEDANG: segitiga (25→50→75)
    mu_sedang = triangular_mf(value, 25, 50, 75)

    # Fungsi TINGGI: trapezoid kanan (50→100→100)
    if value >= 100:
        mu_tinggi = 1.0
    elif value >= 75:
        mu_tinggi = 1.0
    elif value >= 50:
        mu_tinggi = (value - 50) / 25.0
    else:
        mu_tinggi = 0.0

    return {
        "rendah": round(mu_rendah, 4),
        "sedang": round(mu_sedang, 4),
        "tinggi": round(mu_tinggi, 4)
    }

# ─────────────────────────────────────────────
# OUTPUT MEMBERSHIP FUNCTIONS
# Domain output: [0, 100]
# ─────────────────────────────────────────────

def output_mf_bukan_hoaks(x: float) -> float:
    """BUKAN HOAKS: trapezoid kiri, peak di 0–17, turun ke 35"""
    if x <= 0:
        return 1.0
    elif x <= 17:
        return 1.0
    elif x <= 35:
        return (35 - x) / 18.0
    else:
        return 0.0

def output_mf_perlu_periksa(x: float) -> float:
    """PERLU DIPERIKSA: segitiga 25→50→75"""
    return triangular_mf(x, 25, 50, 75)

def output_mf_hoaks_tinggi(x: float) -> float:
    """HOAKS TINGGI: trapezoid kanan, naik dari 65, peak di 83–100"""
    if x >= 100:
        return 1.0
    elif x >= 83:
        return 1.0
    elif x >= 65:
        return (x - 65) / 18.0
    else:
        return 0.0

# ─────────────────────────────────────────────
# 27 ATURAN FUZZY MAMDANI
# Setiap rule: (label_I1, label_I2, label_I3) → (output_label, output_mf)
# ─────────────────────────────────────────────

RULE_TABLE = {
    # I1 TINGGI
    ("tinggi", "tinggi", "tinggi"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("tinggi", "tinggi", "sedang"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("tinggi", "tinggi", "rendah"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("tinggi", "sedang", "tinggi"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("tinggi", "sedang", "sedang"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("tinggi", "sedang", "rendah"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("tinggi", "rendah", "tinggi"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("tinggi", "rendah", "sedang"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("tinggi", "rendah", "rendah"):  ("perlu_periksa",  output_mf_perlu_periksa),
    # I1 SEDANG
    ("sedang", "tinggi", "tinggi"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("sedang", "tinggi", "sedang"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("sedang", "tinggi", "rendah"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("sedang", "sedang", "tinggi"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("sedang", "sedang", "sedang"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("sedang", "sedang", "rendah"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("sedang", "rendah", "tinggi"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("sedang", "rendah", "sedang"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("sedang", "rendah", "rendah"):  ("bukan_hoaks",    output_mf_bukan_hoaks),
    # I1 RENDAH
    ("rendah", "tinggi", "tinggi"):  ("hoaks_tinggi",   output_mf_hoaks_tinggi),
    ("rendah", "tinggi", "sedang"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("rendah", "tinggi", "rendah"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("rendah", "sedang", "tinggi"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("rendah", "sedang", "sedang"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("rendah", "sedang", "rendah"):  ("bukan_hoaks",    output_mf_bukan_hoaks),
    ("rendah", "rendah", "tinggi"):  ("perlu_periksa",  output_mf_perlu_periksa),
    ("rendah", "rendah", "sedang"):  ("bukan_hoaks",    output_mf_bukan_hoaks),
    ("rendah", "rendah", "rendah"):  ("bukan_hoaks",    output_mf_bukan_hoaks),
}

# ─────────────────────────────────────────────
# DEFUZZIFIKASI — Centroid Method
# ─────────────────────────────────────────────

def defuzzify_centroid(active_rules: list) -> float:
    """
    Metode Centroid: H = Σ(x_k * μ_gabungan(x_k)) / Σ(μ_gabungan(x_k))
    active_rules: list of (alpha, output_mf_func)
    Discretization: 1001 titik pada domain [0, 100]
    """
    if not active_rules:
        return 50.0  # default jika tidak ada rule aktif

    N = 1001
    numerator = 0.0
    denominator = 0.0

    for k in range(N):
        x = k * 100.0 / (N - 1)  # 0.0 → 100.0
        # Agregasi: MAX dari semua rule yang aktif (Mamdani)
        mu_agg = 0.0
        for alpha, mf_func in active_rules:
            clipped = min(alpha, mf_func(x))
            mu_agg = max(mu_agg, clipped)
        numerator += x * mu_agg
        denominator += mu_agg

    if denominator < 1e-9:
        return 50.0  # fallback jika semua nol

    return round(numerator / denominator, 2)

# ─────────────────────────────────────────────
# INDIKATOR 1 — Kredibilitas Sumber
# ─────────────────────────────────────────────

def hitung_i1(teks: str, url: str) -> dict:
    """
    I1 = [(a1 + a2 + a3) / 3] × 100
    Indikasi hoaks = 100 - I1
    """
    teks_lower = teks.lower()
    url_lower = url.lower().strip()

    detail = {}

    # a1 — Apakah ada sumber eksplisit?
    pola_sumber = [
        r'\b(menurut|berdasarkan|dikutip dari|dilansir dari|sumber|kata|ujar|ungkap|sebut)\b',
        r'\b(dilaporkan|diberitakan|disampaikan|menyatakan|mengungkapkan)\b',
        r'(detik|kompas|tribun|tempo|republika|antara|cnn|cnbc|liputan|merdeka|suara|kumparan|tirto)',
        r'(kementerian|kominfo|polri|tni|bpom|kemenkes|pemerintah|presiden|gubernur)',
        r'(profesor|dr\.|ph\.d|peneliti|pakar|ahli|ilmuwan)',
        r'(reuters|bbc|ap news|associated press|afp)',
    ]
    a1 = 0
    for pola in pola_sumber:
        if re.search(pola, teks_lower):
            a1 = 1
            break
    detail["a1_ada_sumber"] = a1

    # a2 — Apakah domain dikenal/terpercaya?
    a2 = 0.0
    if url_lower and url_lower not in ["", "tidak ada", "-", "none"]:
        # Ekstrak domain
        domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/\s]+)', url_lower)
        if domain_match:
            domain = domain_match.group(1)
            if any(domain.endswith(m) or m in domain for m in MEDIA_RESMI_INDONESIA):
                a2 = 1.0
                detail["a2_domain_status"] = "terpercaya"
            else:
                a2 = 0.5
                detail["a2_domain_status"] = "kurang dikenal"
        else:
            a2 = 0.0
            detail["a2_domain_status"] = "tidak dikenali"
    else:
        # Periksa dari dalam teks
        media_match = re.search(
            r'(detik\.com|kompas\.com|tribunnews|tempo\.co|antaranews|liputan6|'
            r'cnnindonesia|republika|beritasatu|mediaindonesia|tirto\.id)',
            teks_lower
        )
        if media_match:
            a2 = 1.0
            detail["a2_domain_status"] = "media resmi terdeteksi di teks"
        else:
            a2 = 0.0
            detail["a2_domain_status"] = "tidak ada URL/domain"

    # a3 — Apakah URL valid?
    a3 = 0
    if url_lower and url_lower not in ["", "tidak ada", "-", "none"]:
        if re.match(r'^https?://', url_lower):
            a3 = 1
            detail["a3_url_valid"] = "URL valid (format benar)"
        elif re.match(r'^www\.', url_lower) or '.' in url_lower:
            a3 = 1
            detail["a3_url_valid"] = "URL valid (domain ditemukan)"
        else:
            a3 = 0
            detail["a3_url_valid"] = "Format URL tidak valid"
    else:
        detail["a3_url_valid"] = "Tidak ada URL"

    i1_kredibilitas = round(((a1 + a2 + a3) / 3.0) * 100, 2)
    i1_indikasi = round(100 - i1_kredibilitas, 2)

    detail["a1"] = a1
    detail["a2"] = a2
    detail["a3"] = a3
    detail["i1_kredibilitas"] = i1_kredibilitas

    return {
        "nilai": i1_indikasi,
        "kredibilitas": i1_kredibilitas,
        "detail": detail
    }

# ─────────────────────────────────────────────
# INDIKATOR 2 — Bahasa & Judul Provokatif
# ─────────────────────────────────────────────

def hitung_i2(teks: str) -> dict:
    """
    I2 = v1*f1* + v2*f2* + v3*f3* + v4*f4* + v5*f5*
    fi* = min(fi / theta_i, 1) × 100
    """
    if not teks.strip():
        return {"nilai": 0.0, "detail": {}}

    kata_list = teks.split()
    total_kata = max(len(kata_list), 1)

    # Kalimat: pisahkan berdasarkan tanda baca kalimat
    kalimat_list = re.split(r'[.!?]+', teks)
    kalimat_list = [k.strip() for k in kalimat_list if k.strip()]
    total_kalimat = max(len(kalimat_list), 1)

    detail = {}

    # f1 — CAPSLOCK: kata semua huruf besar (≥2 karakter, bukan angka)
    kata_capslock = [k for k in kata_list if len(k) >= 2 and k.isupper() and k.isalpha()]
    f1_mentah = len(kata_capslock) / total_kata
    theta1 = 0.05
    f1_norm = min(f1_mentah / theta1, 1.0) * 100
    detail["f1_capslock"] = {
        "jumlah": len(kata_capslock),
        "mentah": round(f1_mentah, 4),
        "normalized": round(f1_norm, 2),
        "contoh": kata_capslock[:5]
    }

    # f2 — Tanda baca berlebihan (!! atau ??)
    pola_tanda_berlebih = re.findall(r'[!?]{2,}', teks)
    f2_mentah = len(pola_tanda_berlebih) / total_kalimat
    theta2 = 0.20
    f2_norm = min(f2_mentah / theta2, 1.0) * 100
    detail["f2_tanda_baca"] = {
        "jumlah": len(pola_tanda_berlebih),
        "mentah": round(f2_mentah, 4),
        "normalized": round(f2_norm, 2)
    }

    # f3 — Kata emosional
    teks_lower = teks.lower()
    temuan_emosi = [k for k in KATA_EMOSIONAL if k in teks_lower]
    # Hitung frekuensi total kemunculan
    jumlah_emosi = sum(teks_lower.count(k) for k in temuan_emosi)
    f3_mentah = jumlah_emosi / total_kata
    theta3 = 0.05
    f3_norm = min(f3_mentah / theta3, 1.0) * 100
    detail["f3_emosional"] = {
        "jumlah": jumlah_emosi,
        "kata_ditemukan": list(set(temuan_emosi))[:8],
        "mentah": round(f3_mentah, 4),
        "normalized": round(f3_norm, 2)
    }

    # f4 — Kata hasutan
    temuan_hasutan = [k for k in KATA_HASUTAN if k in teks_lower]
    jumlah_hasutan = sum(teks_lower.count(k) for k in temuan_hasutan)
    f4_mentah = jumlah_hasutan / total_kalimat
    theta4 = 0.25
    f4_norm = min(f4_mentah / theta4, 1.0) * 100
    detail["f4_hasutan"] = {
        "jumlah": jumlah_hasutan,
        "kata_ditemukan": list(set(temuan_hasutan))[:8],
        "mentah": round(f4_mentah, 4),
        "normalized": round(f4_norm, 2)
    }

    # f5 — Kata hiperbola/bombastis
    temuan_hiperbola = [k for k in KATA_HIPERBOLA if k in teks_lower]
    jumlah_hiperbola = sum(teks_lower.count(k) for k in temuan_hiperbola)
    f5_mentah = jumlah_hiperbola / total_kalimat
    theta5 = 0.375
    f5_norm = min(f5_mentah / theta5, 1.0) * 100
    detail["f5_hiperbola"] = {
        "jumlah": jumlah_hiperbola,
        "kata_ditemukan": list(set(temuan_hiperbola))[:8],
        "mentah": round(f5_mentah, 4),
        "normalized": round(f5_norm, 2)
    }

    # Weighted sum
    bobot = {"v1": 0.20, "v2": 0.15, "v3": 0.30, "v4": 0.20, "v5": 0.15}
    i2 = round(
        bobot["v1"] * f1_norm +
        bobot["v2"] * f2_norm +
        bobot["v3"] * f3_norm +
        bobot["v4"] * f4_norm +
        bobot["v5"] * f5_norm,
        2
    )

    detail["bobot"] = bobot
    detail["total_kata"] = total_kata
    detail["total_kalimat"] = total_kalimat

    return {"nilai": i2, "detail": detail}

# ─────────────────────────────────────────────
# INDIKATOR 3 — Kelengkapan Informasi (5W+1H)
# ─────────────────────────────────────────────

def hitung_i3(teks: str) -> dict:
    """
    K = Σ(uj * cj) × 100
    I3 = 100 - K
    """
    teks_lower = teks.lower()
    detail = {}

    # WHO — Nama orang / lembaga spesifik
    pola_who = [
        r'\b(menteri|presiden|gubernur|walikota|bupati|kapolri|panglima|direktur|kepala)\b',
        r'\b(prabowo|jokowi|megawati|anies|ahok|ganjar)\b',
        r'\b(polri|tni|kemenkes|kominfo|bpom|kpu|mahkamah|dpr|mpr)\b',
        r'\b(bapak|ibu|pak|bu)\s+\w+\b',
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
    ]
    who_score = 0.0
    for p in pola_who:
        if re.search(p, teks, re.IGNORECASE):
            who_score = 1.0
            break
    if who_score == 0.0:
        nama_umum = re.search(r'\b(warga|masyarakat|orang|seseorang|pihak|mereka)\b', teks_lower)
        if nama_umum:
            who_score = 0.5
    detail["who"] = {"skor": who_score, "bobot": 0.20}

    # WHAT — Subjek kejadian spesifik
    pola_what = [
        r'\b(gempa|banjir|kebakaran|kecelakaan|ledakan|tsunami|longsor)\b',
        r'\b(virus|vaksin|obat|penyakit|wabah|pandemi|kasus)\b',
        r'\b(korupsi|penangkapan|pemilu|sidang|vonis|putusan)\b',
        r'\b(kenaikan|penurunan|perubahan|pembangunan|penghapusan)\b',
        r'\b(terjadi|ditemukan|dilaporkan|terdeteksi|ditangkap|diamankan)\b',
    ]
    what_score = 0.0
    for p in pola_what:
        if re.search(p, teks_lower):
            what_score = 1.0
            break
    if what_score == 0.0 and len(teks.split()) > 20:
        what_score = 0.5
    detail["what"] = {"skor": what_score, "bobot": 0.20}

    # WHEN — Tanggal / waktu spesifik
    pola_when = [
        r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b',
        r'\b\d{1,2}\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+\d{4}\b',
        r'\b(senin|selasa|rabu|kamis|jumat|sabtu|minggu),?\s+\d{1,2}\b',
        r'\b(pagi|siang|sore|malam|subuh)\s+(ini|tadi|kemarin)\b',
        r'\b(hari ini|kemarin|besok|minggu lalu|bulan lalu|tahun lalu)\b',
        r'\b(jam|pukul)\s+\d{1,2}[\.:]\d{2}\b',
        r'\(\d{1,2}/\d{1,2}/?\d{0,4}\)',
    ]
    when_score = 0.0
    for p in pola_when:
        if re.search(p, teks_lower):
            when_score = 1.0
            break
    detail["when"] = {"skor": when_score, "bobot": 0.20}

    # WHERE — Lokasi spesifik
    pola_where = [
        r'\b(jakarta|surabaya|bandung|medan|semarang|makassar|yogyakarta|bali|papua)\b',
        r'\b(indonesia|indonesia)\b',
        r'\b(jalan|jl\.|kelurahan|kecamatan|kabupaten|kota|provinsi|desa|kampung)\b',
        r'\b(rumah sakit|rs\s+\w+|puskesmas|kantor|gedung|istana|kementerian)\b',
        r'\b(barat|timur|utara|selatan|tengah)\s+(indonesia|jawa|kalimantan|sumatra|sulawesi)\b',
    ]
    where_score = 0.0
    for p in pola_where:
        if re.search(p, teks_lower):
            where_score = 1.0
            break
    detail["where"] = {"skor": where_score, "bobot": 0.15}

    # WHY — Penjelasan sebab / alasan
    pola_why = [
        r'\b(karena|sebab|lantaran|dikarenakan|akibat|disebabkan|mengakibatkan)\b',
        r'\b(akibat dari|dampak dari|berkaitan dengan|sehubungan dengan)\b',
        r'\b(alasan|tujuan|motif|maksud|latar belakang)\b',
    ]
    why_score = 0.0
    for p in pola_why:
        if re.search(p, teks_lower):
            why_score = 1.0
            break
    detail["why"] = {"skor": why_score, "bobot": 0.15}

    # HOW — Penjelasan proses/mekanisme
    pola_how = [
        r'\b(dengan cara|melalui|menggunakan|secara|langkah|prosedur|mekanisme)\b',
        r'\b(caranya|prosesnya|metodanya|tekniknya|strateginya)\b',
        r'\b(pertama|kedua|ketiga|selanjutnya|kemudian|lalu|akhirnya)\b',
    ]
    how_score = 0.0
    for p in pola_how:
        if re.search(p, teks_lower):
            how_score = 1.0
            break
    detail["how"] = {"skor": how_score, "bobot": 0.10}

    # Hitung K dan I3
    K = round(
        0.20 * detail["who"]["skor"] +
        0.20 * detail["what"]["skor"] +
        0.20 * detail["when"]["skor"] +
        0.15 * detail["where"]["skor"] +
        0.15 * detail["why"]["skor"] +
        0.10 * detail["how"]["skor"],
        4
    ) * 100

    i3 = round(100 - K, 2)

    return {"nilai": i3, "kelengkapan_k": round(K, 2), "detail": detail}

# ─────────────────────────────────────────────
# MESIN FUZZY MAMDANI UTAMA
# ─────────────────────────────────────────────

def fuzzy_mamdani(i1_val: float, i2_val: float, i3_val: float) -> dict:
    """
    Pipeline lengkap Fuzzy Mamdani:
    1. Fuzzifikasi
    2. Rule Evaluation (27 rules)
    3. Agregasi
    4. Defuzzifikasi (Centroid)
    """
    # Step 1: Fuzzifikasi
    mu_i1 = fuzzify(i1_val)
    mu_i2 = fuzzify(i2_val)
    mu_i3 = fuzzify(i3_val)

    # Step 2: Evaluasi semua 27 rules
    active_rules = []
    rule_log = []

    labels = ["rendah", "sedang", "tinggi"]

    for l1 in labels:
        for l2 in labels:
            for l3 in labels:
                rule_key = (l1, l2, l3)
                if rule_key not in RULE_TABLE:
                    continue

                output_label, output_mf = RULE_TABLE[rule_key]
                alpha = min(mu_i1[l1], mu_i2[l2], mu_i3[l3])

                rule_log.append({
                    "kondisi": f"I1={l1.upper()}, I2={l2.upper()}, I3={l3.upper()}",
                    "output": output_label,
                    "alpha": round(alpha, 4)
                })

                if alpha > 1e-6:  # Hanya masukkan rule yang aktif
                    active_rules.append((alpha, output_mf))

    # Step 3 + 4: Agregasi & Defuzzifikasi
    H = defuzzify_centroid(active_rules)

    # Klasifikasi akhir
    if H < 35:
        kategori = "BUKAN HOAKS"
        warna = "hijau"
        rekomendasi = "Konten ini relatif aman dikonsumsi. Tetap lakukan verifikasi mandiri ke sumber resmi untuk kepastian."
        emoji = "✅"
    elif H < 65:
        kategori = "PERLU DIPERIKSA"
        warna = "kuning"
        rekomendasi = "Ada beberapa indikasi yang perlu ditelusuri. Jangan disebarkan sebelum memverifikasi ke sumber terpercaya seperti Kominfo, Turnbackhoax.id, atau media nasional."
        emoji = "⚠️"
    else:
        kategori = "KEMUNGKINAN HOAKS TINGGI"
        warna = "merah"
        rekomendasi = "Konten ini memiliki banyak ciri khas hoaks. JANGAN disebarkan. Laporkan ke Kominfo (aduankonten.id) atau cek di Turnbackhoax.id dan CekFakta.com."
        emoji = "🚨"

    return {
        "skor_h": H,
        "kategori": kategori,
        "warna": warna,
        "rekomendasi": rekomendasi,
        "emoji": emoji,
        "fuzzifikasi": {
            "I1": mu_i1,
            "I2": mu_i2,
            "I3": mu_i3
        },
        "rules_aktif": len([r for r in rule_log if r["alpha"] > 1e-6]),
        "rules_log": sorted(rule_log, key=lambda x: x["alpha"], reverse=True)[:10]
    }

# ─────────────────────────────────────────────
# ENDPOINT UTAMA
# ─────────────────────────────────────────────

@app.route("/analisis", methods=["POST"])
def analisis():
    try:
        data = request.get_json(force=True)
        teks = str(data.get("teks", "")).strip()
        url = str(data.get("url", "")).strip()

        # Validasi input
        if not teks:
            return jsonify({"error": "Teks tidak boleh kosong."}), 400

        if len(teks) < 10:
            return jsonify({"error": "Teks terlalu pendek. Masukkan minimal 10 karakter."}), 400

        if len(teks) > 10000:
            teks = teks[:10000]  # Batasi aman

        # Hitung 3 indikator
        hasil_i1 = hitung_i1(teks, url)
        hasil_i2 = hitung_i2(teks)
        hasil_i3 = hitung_i3(teks)

        i1_val = hasil_i1["nilai"]
        i2_val = hasil_i2["nilai"]
        i3_val = hasil_i3["nilai"]

        # Proses Fuzzy Mamdani
        hasil_fuzzy = fuzzy_mamdani(i1_val, i2_val, i3_val)

        # Susun respons lengkap
        response = {
            "status": "ok",
            "input": {
                "panjang_teks": len(teks),
                "ada_url": bool(url and url not in ["-", "tidak ada"])
            },
            "indikator": {
                "I1": {
                    "label": "Kredibilitas Sumber",
                    "nilai_indikasi": i1_val,
                    "nilai_kredibilitas": hasil_i1["kredibilitas"],
                    "detail": hasil_i1["detail"]
                },
                "I2": {
                    "label": "Bahasa & Judul Provokatif",
                    "nilai": i2_val,
                    "detail": hasil_i2["detail"]
                },
                "I3": {
                    "label": "Kelengkapan Informasi (5W+1H)",
                    "nilai_indikasi": i3_val,
                    "kelengkapan_k": hasil_i3["kelengkapan_k"],
                    "detail": hasil_i3["detail"]
                }
            },
            "fuzzy": hasil_fuzzy,
            "hasil": {
                "skor": hasil_fuzzy["skor_h"],
                "kategori": hasil_fuzzy["kategori"],
                "warna": hasil_fuzzy["warna"],
                "rekomendasi": hasil_fuzzy["rekomendasi"],
                "emoji": hasil_fuzzy["emoji"]
            }
        }

        return jsonify(response), 200

    except Exception as e:
        # Tangkap semua error agar tidak crash saat demo
        app.logger.error(f"Error: {traceback.format_exc()}")
        return jsonify({
            "error": "Terjadi kesalahan pada server. Silakan coba lagi.",
            "detail": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "SYD Score Backend aktif", "versi": "1.0.0"}), 200


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "sistem": "SYD Score — Scoring-based Hoax Yield Detection",
        "deskripsi": "Backend Fuzzy Logic Mamdani untuk deteksi hoaks",
        "endpoint": "/analisis (POST)",
        "versi": "1.0.0"
    }), 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
