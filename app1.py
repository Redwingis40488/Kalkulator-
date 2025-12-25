from flask import Flask, request, jsonify, render_template_string
import sympy as sp
import threading, webbrowser, time
import logging
import math
import io
import base64

# =======================
# MATPLOTLIB SETUP (TERMUX/SERVER SAFE)
# =======================
import matplotlib
matplotlib.use('Agg') # Wajib agar jalan tanpa GUI window (Termux friendly)
import matplotlib.pyplot as plt
import numpy as np

# =======================
# CONFIG & LOGGING
# =======================
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# =======================
# SYMPY INIT
# =======================
x, y = sp.symbols('x y')

# =======================
# HELPER & FORMATTING
# =======================

class Color:
    PURPLE = '#9333ea'
    CYAN = '#06b6d4'
    BLUE = '#3b82f6'
    GREEN = '#10b981'
    YELLOW = '#eab308'
    RED = '#ef4444'

def fnum(val):
    """Format angka untuk tampilan"""
    try:
        if val is None:
            return "Tidak ada"
        val_float = float(val.evalf())
        if abs(val_float - round(val_float)) < 1e-9:
            return str(int(round(val_float)))
        return f"{val_float:.2f}"
    except:
        return str(val)

def get_val(sympy_val):
    """Helper untuk mengambil float dari sympy"""
    return float(sympy_val.evalf())

def to_rad(deg):
    return deg * sp.pi / 180

# =======================
# VISUALIZATION ENGINE (NEW)
# =======================
class Plotter:
    @staticmethod
    def create_triangle_image(a, b, c, A_deg, B_deg, C_deg, title="Visualisasi Segitiga"):
        """
        Membuat plot segitiga berdasarkan 3 sisi dan 3 sudut.
        Titik A di (0,0), B di (c,0).
        """
        try:
            # Konversi ke float native Python
            side_a = float(a)
            side_b = float(b)
            side_c = float(c)
            angle_A = float(A_deg) * np.pi / 180
            # angle_B tidak dipakai untuk koordinat, tapi untuk label
            
            # Koordinat Titik
            # A = (0, 0)
            # B = (c, 0)
            # C = (b * cos(A), b * sin(A))
            
            Ax, Ay = 0, 0
            Bx, By = side_c, 0
            Cx = side_b * np.cos(angle_A)
            Cy = side_b * np.sin(angle_A)
            
            # Setup Plot
            fig, ax = plt.subplots(figsize=(6, 4.5)) # Ukuran pas untuk web
            
            # Gambar Segitiga
            x_vals = [Ax, Bx, Cx, Ax]
            y_vals = [Ay, By, Cy, Ay]
            
            # Fill warna gradasi (simulasi dengan alpha)
            ax.fill(x_vals, y_vals, color='#0ea5e9', alpha=0.1)
            ax.plot(x_vals, y_vals, color='#0ea5e9', linewidth=2, marker='o', markersize=6)
            
            # Label Titik
            offset = side_c * 0.05
            ax.text(Ax - offset, Ay, f'A\n{fnum(A_deg)}°', fontsize=10, ha='right', color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
            ax.text(Bx + offset, By, f'B\n{fnum(B_deg)}°', fontsize=10, ha='left', color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
            ax.text(Cx, Cy + offset, f'C\n{fnum(C_deg)}°', fontsize=10, ha='center', color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Label Sisi
            ax.text((Ax+Bx)/2, Ay - offset/2, f'c = {fnum(side_c)}', ha='center', va='top', color='#94a3b8', fontsize=9)
            ax.text((Bx+Cx)/2, (By+Cy)/2, f'a = {fnum(side_a)}', ha='left', va='bottom', color='#94a3b8', fontsize=9)
            ax.text((Ax+Cx)/2, (Ay+Cy)/2, f'b = {fnum(side_b)}', ha='right', va='bottom', color='#94a3b8', fontsize=9)
            
            # Styling Axis
            ax.set_aspect('equal')
            ax.axis('off') # Hilangkan sumbu X/Y biar bersih
            
            # Judul Kecil di dalam plot
            plt.title(title, color='white', fontsize=10, pad=10)

            # Simpan ke Buffer
            buf = io.BytesIO()
            # Set background transparan agar menyatu dengan Glassmorphism
            plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
            plt.close(fig)
            buf.seek(0)
            
            # Encode Base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return img_base64
        except Exception as e:
            print(f"Plot Error: {e}")
            return None

# =======================
# ENGINE GEOMETRI (ASLI - TIDAK UBAH)
# =======================
class GeoEngine:
    @staticmethod
    def get_matrix(mode, param=None):
        if mode == 'x': return sp.Matrix([[1, 0], [0, -1]])
        if mode == 'y': return sp.Matrix([[-1, 0], [0, 1]])
        if mode == 'yx': return sp.Matrix([[0, 1], [1, 0]])
        if mode == 'y-x': return sp.Matrix([[0, -1], [-1, 0]])
        if mode == 'origin': return sp.Matrix([[-1, 0], [0, -1]])
        if mode == 'rot':
            theta = to_rad(param)
            return sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])
        if mode == 'dil':
            return sp.Matrix([[param, 0], [0, param]])
        return sp.eye(2)

    @staticmethod
    def get_matrix_homogen_3x3(mode, param=None, tx=0, ty=0):
        if mode == 'trans':
            return sp.Matrix([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        m2 = GeoEngine.get_matrix(mode, param)
        return sp.Matrix([
            [m2[0,0], m2[0,1], 0],
            [m2[1,0], m2[1,1], 0],
            [0,       0,       1]
        ])

    @staticmethod
    def translasi(point, T):
        px, py = point
        tx, ty = T
        res = (px + tx, py + ty)
        steps = [
            f"Titik awal P({px}, {py})",
            f"Vektor geser T({tx}, {ty})",
            f"x' = {px} + {tx} = {px+tx}",
            f"y' = {py} + {ty} = {py+ty}"
        ]
        return res, steps

    @staticmethod
    def translasi_homogen(point, T):
        px, py = point
        tx, ty = T
        mat = GeoEngine.get_matrix_homogen_3x3('trans', tx=tx, ty=ty)
        p_vec = sp.Matrix([px, py, 1])
        res_vec = mat * p_vec
        res = (res_vec[0], res_vec[1])
        steps = [
            f"Mengubah P({px}, {py}) ke koordinat homogen: Matrix([x, y, 1])",
            f"Matriks Translasi 3x3: [[1,0,{tx}],[0,1,{ty}],[0,0,1]]",
            f"Hasil perkalian: [{res_vec[0]}, {res_vec[1]}, 1]"
        ]
        return res, steps, mat

    @staticmethod
    def invers_transformasi(mode, param=None):
        mat = GeoEngine.get_matrix(mode, param)
        try:
            mat_inv = mat.inv()
            steps = [
                f"Matriks Asal M:\n{mat}",
                f"Determinan = {fnum(mat.det())}",
                "Invers M⁻¹ = 1/det(M) × Adjoin(M)"
            ]
            return mat_inv, steps
        except:
            return None, ["Matriks singular, tidak punya invers."]

    @staticmethod
    def refleksi(point, mode):
        px, py = point
        mat = GeoEngine.get_matrix(mode)
        p_vec = sp.Matrix([px, py])
        res_vec = mat * p_vec
        res = (res_vec[0], res_vec[1])
        steps = [
            f"Titik awal P({px}, {py})",
            f"Matriks refleksi: {mat}",
            f"P' = Matriks × P = {res}"
        ]
        return res, steps, mat

    @staticmethod
    def rotasi(point, angle_deg, center=(0,0)):
        px, py = point
        cx, cy = center
        mat = GeoEngine.get_matrix('rot', angle_deg)
        
        if cx == 0 and cy == 0:
            p_vec = sp.Matrix([px, py])
            res_vec = mat * p_vec
            steps = [f"Rotasi pusat (0,0) sudut {angle_deg}°"]
        else:
            p_vec = sp.Matrix([px - cx, py - cy])
            temp_vec = mat * p_vec
            res_vec = temp_vec + sp.Matrix([cx, cy])
            steps = [f"Rotasi pusat ({cx},{cy}) sudut {angle_deg}°: Geser-Putar-Geser"]

        res = (res_vec[0], res_vec[1])
        return res, steps, mat

    @staticmethod
    def dilatasi(point, factor, center=(0,0)):
        px, py = point
        cx, cy = center
        k = factor
        mat = GeoEngine.get_matrix('dil', k)
        
        if cx == 0 and cy == 0:
            p_vec = sp.Matrix([px, py])
            res_vec = mat * p_vec
            steps = [f"Dilatasi pusat (0,0) faktor k={k}"]
        else:
            p_vec = sp.Matrix([px - cx, py - cy])
            temp_vec = mat * p_vec
            res_vec = temp_vec + sp.Matrix([cx, cy])
            steps = [f"Dilatasi pusat ({cx},{cy}) faktor k={k}: (x'-{cx}) = {k}·(x-{cx})"]

        res = (res_vec[0], res_vec[1])
        return res, steps, mat

# =======================
# ENGINE TRIGONOMETRI (UPDATED WITH PLOT CALLS)
# =======================
class TrigEngine:
    @staticmethod
    def luas_segitiga(a, b, angle_C):
        rad_C = to_rad(angle_C)
        val_sin = sp.sin(rad_C)
        res = 0.5 * a * b * val_sin
        steps = [
            f"sin({angle_C}°) = {fnum(val_sin)}",
            f"L = ½ × {a} × {b} × {fnum(val_sin)}",
            f"L = {fnum(res)}"
        ]
        
        # Hitung sisi c untuk visualisasi
        c_sq = a**2 + b**2 - (2 * a * b * sp.cos(rad_C))
        c_res = sp.sqrt(c_sq)
        
        # Hitung sudut lain (Sines)
        # a/sinA = c/sinC
        rad_A = sp.asin(a * val_sin / c_res)
        deg_A = rad_A * 180 / sp.pi
        deg_B = 180 - angle_C - deg_A
        
        img = Plotter.create_triangle_image(a, b, c_res, deg_A, deg_B, angle_C)
        return res, steps, img

    @staticmethod
    def aturan_sinus_ambigu(sisi_a, sisi_b, sudut_A):
        rad_A = to_rad(sudut_A)
        sin_A = sp.sin(rad_A)
        h = sisi_b * sin_A
        
        steps = [
            f"Diketahui: a={sisi_a}, b={sisi_b}, A={sudut_A}°",
            f"Tinggi h = b × sin A = {sisi_b} × {fnum(sin_A)} = {fnum(h)}"
        ]
        
        val_a = float(sisi_a.evalf())
        val_b = float(sisi_b.evalf())
        val_h = float(h.evalf())
        
        images = []
        
        if val_a < val_h:
            steps.append(f"Karena a < h ({val_a} < {fnum(val_h)}), tidak ada segitiga")
            return [], steps, "Tidak ada solusi (0 Segitiga)", []
            
        elif abs(val_a - val_h) < 1e-9:
            deg_B = 90
            deg_C = 180 - float(sudut_A) - 90
            # Cari sisi c
            # c/sinC = a/sinA
            val_c = (val_a * math.sin(math.radians(deg_C))) / math.sin(math.radians(float(sudut_A)))
            
            steps.append(f"Karena a = h, segitiga siku-siku di B")
            img = Plotter.create_triangle_image(val_a, val_b, val_c, float(sudut_A), deg_B, deg_C)
            return [deg_B], steps, "1 Solusi (Siku-siku)", [img]
            
        elif val_a >= val_b:
            val_sin_B = (sisi_b * sin_A) / sisi_a
            rad_B = sp.asin(val_sin_B)
            deg_B = float(rad_B * 180 / sp.pi)
            deg_C = 180 - float(sudut_A) - deg_B
            val_c = (val_a * math.sin(math.radians(deg_C))) / math.sin(math.radians(float(sudut_A)))

            steps.append(f"Karena a ≥ b, hanya ada 1 segitiga")
            steps.append(f"B = {fnum(deg_B)}°")
            
            img = Plotter.create_triangle_image(val_a, val_b, val_c, float(sudut_A), deg_B, deg_C)
            return [deg_B], steps, "1 Solusi", [img]
            
        else:
            # Kasus Ambigu (2 Segitiga)
            val_sin_B = (sisi_b * sin_A) / sisi_a
            rad_B1 = sp.asin(val_sin_B)
            deg_B1 = float(rad_B1 * 180 / sp.pi)
            deg_B2 = 180 - deg_B1
            
            deg_C1 = 180 - float(sudut_A) - deg_B1
            deg_C2 = 180 - float(sudut_A) - deg_B2
            
            val_c1 = (val_a * math.sin(math.radians(deg_C1))) / math.sin(math.radians(float(sudut_A)))
            val_c2 = (val_a * math.sin(math.radians(deg_C2))) / math.sin(math.radians(float(sudut_A)))
            
            steps.append(f"Karena h < a < b, ada 2 kemungkinan")
            
            img1 = Plotter.create_triangle_image(val_a, val_b, val_c1, float(sudut_A), deg_B1, deg_C1, "Solusi 1 (Lancip)")
            img2 = Plotter.create_triangle_image(val_a, val_b, val_c2, float(sudut_A), deg_B2, deg_C2, "Solusi 2 (Tumpul)")
            
            return [deg_B1, deg_B2], steps, "2 Solusi (Ambigu)", [img1, img2]

    @staticmethod
    def aturan_cosinus(a=None, b=None, c=None, angle_C=None):
        if c is None:
            # Cari Sisi
            rad_C = to_rad(angle_C)
            val_cos = sp.cos(rad_C)
            c_sq = a**2 + b**2 - (2 * a * b * val_cos)
            res = sp.sqrt(c_sq)
            steps = [
                f"c² = a² + b² - 2ab·cos C",
                f"c = √{fnum(c_sq)} = {fnum(res)}"
            ]
            
            # Hitung sudut A untuk plot
            # a^2 = b^2 + c^2 - 2bc cosA
            val_a = get_val(a)
            val_b = get_val(b)
            val_c = get_val(res)
            val_ang_C = get_val(angle_C)
            
            cos_A = (val_b**2 + val_c**2 - val_a**2)/(2*val_b*val_c)
            # Clamp value domain cosinus
            cos_A = max(-1, min(1, cos_A)) 
            deg_A = math.degrees(math.acos(cos_A))
            deg_B = 180 - val_ang_C - deg_A
            
            img = Plotter.create_triangle_image(val_a, val_b, val_c, deg_A, deg_B, val_ang_C)
            return res, steps, img
            
        elif angle_C is None:
            # Cari Sudut
            val_cos = (a**2 + b**2 - c**2) / (2 * a * b)
            res = sp.acos(val_cos) * 180 / sp.pi
            steps = [
                f"cos C = (a² + b² - c²) / 2ab",
                f"C = {fnum(res)}°"
            ]
            
            val_a = get_val(a)
            val_b = get_val(b)
            val_c = get_val(c)
            val_res_C = get_val(res)
            
            # Hitung sudut A
            cos_A = (val_b**2 + val_c**2 - val_a**2)/(2*val_b*val_c)
            cos_A = max(-1, min(1, cos_A))
            deg_A = math.degrees(math.acos(cos_A))
            deg_B = 180 - val_res_C - deg_A
            
            img = Plotter.create_triangle_image(val_a, val_b, val_c, deg_A, deg_B, val_res_C)
            return res, steps, img

# =======================
# FRONTEND (HTML/CSS/JS)
# =======================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Smart Geometry & Trigonometry PRO</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=JetBrains+Mono&display=swap');

        :root {
            --bg: #050810;
            --glass: rgba(20, 25, 40, 0.7);
            --accent: #0ea5e9;
            --accent-light: #38bdf8;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --border: rgba(255, 255, 255, 0.1);
        }

        body {
            margin: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(14, 165, 233, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(129, 140, 248, 0.15) 0%, transparent 40%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 1100px;
            display: grid;
            grid-template-columns: 1fr 1.3fr;
            gap: 25px;
            animation: fadeIn 0.8s ease-out;
        }

        @media (max-width: 900px) {
            .container { grid-template-columns: 1fr; }
        }

        /* --- LEFT PANEL: INPUT --- */
        .panel-input {
            background: var(--glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
        }

        .header h1 { 
            font-size: 1.6rem; 
            margin: 0; 
            font-weight: 800; 
            background: linear-gradient(135deg, var(--accent), #9333ea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p { color: var(--text-dim); font-size: 0.85rem; margin-top: 5px; margin-bottom: 25px;}

        .module-tab {
            display: flex;
            background: rgba(0,0,0,0.3);
            padding: 4px;
            border-radius: 12px;
            margin-bottom: 20px;
        }

        .tab-btn {
            flex: 1;
            padding: 10px;
            border: none;
            background: transparent;
            color: var(--text-dim);
            cursor: pointer;
            font-weight: 600;
            border-radius: 10px;
            transition: all 0.3s;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, var(--accent), #6366f1);
            color: white;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        }

        label { 
            display: block; 
            font-size: 0.75rem; 
            color: var(--accent-light); 
            font-weight: 700; 
            margin-bottom: 6px; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        input, textarea, select {
            width: 100%;
            background: rgba(0,0,0,0.4);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            color: white;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 15px;
            outline: none;
            box-sizing: border-box; 
            transition: all 0.3s;
            font-size: 0.9rem;
        }

        input:focus, select:focus { 
            border-color: var(--accent); 
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2); 
        }

        .input-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .btn-calc {
            width: 100%;
            background: linear-gradient(135deg, var(--accent), #6366f1);
            color: white;
            border: none;
            padding: 16px;
            border-radius: 12px;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 10px 20px -5px rgba(14, 165, 233, 0.4);
            transition: all 0.3s;
            font-size: 1rem;
            margin-top: auto;
        }

        .btn-calc:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 15px 25px -5px rgba(14, 165, 233, 0.5);
        }

        /* --- RIGHT PANEL: OUTPUT --- */
        .panel-output {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* CARD HASIL TEKS */
        .card-result {
            background: var(--glass);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 25px;
            min-height: 120px;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }

        /* CARD VISUALISASI (BARU) */
        .card-visual {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 20px;
            display: none; /* Hidden by default */
            flex-direction: column;
            align-items: center;
            overflow: hidden;
            animation: slideDown 0.5s ease-out;
        }
        
        .visual-img {
            max-width: 100%;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            transition: transform 0.3s;
        }
        
        .visual-img:hover {
            transform: scale(1.02);
        }

        /* CARD STEP */
        .card-explanation {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 25px;
            font-size: 0.9rem;
            line-height: 1.6;
            flex-grow: 1;
        }

        .math-display { 
            font-size: 1.4rem; 
            text-align: center;
            width: 100%;
        }

        .hidden { display: none; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideDown { from { opacity: 0; height: 0; } to { opacity: 1; height: auto; } }

        .step-item {
            padding: 10px 0;
            border-bottom: 1px solid var(--border);
            color: var(--text-dim);
        }
        
        .step-item:last-child { border-bottom: none; }
        .step-item b { color: var(--text); display: block; margin-bottom: 4px;}
        
        .status-box {
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-weight: 600;
            background: rgba(14, 165, 233, 0.1);
            border-left: 3px solid var(--accent);
            font-size: 0.9rem;
        }
        
        .status-box.warning { background: rgba(234, 179, 8, 0.1); border-left-color: #eab308; }
        
        .visual-title {
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }
        
        .visual-title span { font-weight: 700; color: var(--accent-light); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;}
        .badge { background: var(--accent); color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 800; }
    </style>
</head>
<body>

<div class="container">
    <div class="panel-input">
        <div class="header">
            <h1>GeoTrig Ultimate</h1>
            <p>Kalkulator Cerdas & Visualisasi Presisi</p>
        </div>

        <div class="module-tab">
            <button class="tab-btn active" id="tab-geo" onclick="switchTab('geo')">Geometri</button>
            <button class="tab-btn" id="tab-trig" onclick="switchTab('trig')">Trigonometri</button>
        </div>

        <div id="form-geo">
            <label>Jenis Transformasi</label>
            <select id="geo-op" onchange="updateGeoForm()">
                <option value="translasi">Translasi (Vektor)</option>
                <option value="translasi_homogen">Translasi (Matriks 3x3)</option>
                <option value="refleksi">Refleksi</option>
                <option value="rotasi">Rotasi</option>
                <option value="dilatasi">Dilatasi</option>
                <option value="invers">Invers Matriks</option>
            </select>

            <div id="geo-inputs">
                <div class="input-group">
                    <div><label>Titik P(x)</label><input type="text" id="geo-px" placeholder="2" value="2"></div>
                    <div><label>Titik P(y)</label><input type="text" id="geo-py" placeholder="3" value="3"></div>
                </div>

                <div id="geo-vector" class="input-group">
                    <div><label>Vektor T(x)</label><input type="text" id="geo-tx" placeholder="1" value="1"></div>
                    <div><label>Vektor T(y)</label><input type="text" id="geo-ty" placeholder="2" value="2"></div>
                </div>

                <div id="geo-refleksi" class="hidden">
                    <label>Garis Refleksi</label>
                    <select id="geo-mode">
                        <option value="x">Sumbu X</option>
                        <option value="y">Sumbu Y</option>
                        <option value="yx">Garis y = x</option>
                        <option value="y-x">Garis y = -x</option>
                        <option value="origin">Titik Asal</option>
                    </select>
                </div>

                <div id="geo-rotasi" class="hidden">
                    <label>Sudut (derajat)</label>
                    <input type="text" id="geo-angle" placeholder="90" value="90">
                    <div class="input-group">
                        <div><label>Pusat X</label><input type="text" id="geo-cx" placeholder="0" value="0"></div>
                        <div><label>Pusat Y</label><input type="text" id="geo-cy" placeholder="0" value="0"></div>
                    </div>
                </div>

                <div id="geo-dilatasi" class="hidden">
                    <label>Faktor Skala (k)</label>
                    <input type="text" id="geo-factor" placeholder="2" value="2">
                    <div class="input-group">
                        <div><label>Pusat X</label><input type="text" id="geo-dcx" placeholder="0" value="0"></div>
                        <div><label>Pusat Y</label><input type="text" id="geo-dcy" placeholder="0" value="0"></div>
                    </div>
                </div>

                <div id="geo-invers" class="hidden">
                    <label>Tipe</label>
                    <select id="geo-inv-type">
                        <option value="rot">Rotasi</option>
                        <option value="dil">Dilatasi</option>
                    </select>
                    <label style="margin-top:10px">Parameter</label>
                    <input type="text" id="geo-param" placeholder="Sudut atau Skala" value="90">
                </div>
            </div>
        </div>

        <div id="form-trig" class="hidden">
            <label>Operasi</label>
            <select id="trig-op" onchange="updateTrigForm()">
                <option value="aturan_sinus">Aturan Sinus (Cari Sisi)</option>
                <option value="aturan_sinus_ambigu">Aturan Sinus (Analisis Ambigu)</option>
                <option value="aturan_cosinus">Aturan Cosinus</option>
                <option value="luas_segitiga">Luas Segitiga</option>
            </select>

            <div id="trig-inputs">
                <div id="trig-sinus">
                    <div class="input-group">
                        <div><label>Sisi b</label><input type="text" id="trig-b" placeholder="5" value="5"></div>
                        <div><label>Sudut A°</label><input type="text" id="trig-A" placeholder="30" value="30"></div>
                    </div>
                    <label>Sudut B°</label><input type="text" id="trig-B" placeholder="45" value="45">
                </div>

                <div id="trig-sinus-ambigu" class="hidden">
                    <div class="input-group">
                        <div><label>Sisi a</label><input type="text" id="trig-a-ambi" placeholder="5" value="5"></div>
                        <div><label>Sisi b</label><input type="text" id="trig-b-ambi" placeholder="7" value="7"></div>
                    </div>
                    <label>Sudut A°</label><input type="text" id="trig-A-ambi" placeholder="30" value="30">
                </div>

                <div id="trig-cosinus" class="hidden">
                    <label>Target</label>
                    <select id="trig-cos-type" onchange="updateCosForm()">
                        <option value="cari_sisi">Cari Sisi c</option>
                        <option value="cari_sudut">Cari Sudut C</option>
                    </select>
                    
                    <div id="trig-cos-sisi" style="margin-top:15px">
                        <div class="input-group">
                            <div><label>Sisi a</label><input type="text" id="trig-a-cos" placeholder="5" value="5"></div>
                            <div><label>Sisi b</label><input type="text" id="trig-b-cos" placeholder="6" value="6"></div>
                        </div>
                        <label>Sudut C°</label><input type="text" id="trig-C-angle" placeholder="60" value="60">
                    </div>
                    
                    <div id="trig-cos-sudut" class="hidden" style="margin-top:15px">
                        <div class="input-group">
                            <div><label>Sisi a</label><input type="text" id="trig-a-cos2" value="5"></div>
                            <div><label>Sisi b</label><input type="text" id="trig-b-cos2" value="6"></div>
                        </div>
                        <label>Sisi c</label><input type="text" id="trig-c-cos2" value="7">
                    </div>
                </div>

                <div id="trig-luas" class="hidden">
                    <div class="input-group">
                        <div><label>Sisi a</label><input type="text" id="trig-a-luas" value="5"></div>
                        <div><label>Sisi b</label><input type="text" id="trig-b-luas" value="6"></div>
                    </div>
                    <label>Sudut C°</label><input type="text" id="trig-C-luas" value="30">
                </div>
            </div>
        </div>

        <button class="btn-calc" onclick="calculate()">HITUNG SEKARANG</button>
    </div>

    <div class="panel-output">
        <div class="card-result">
            <label style="position:absolute; top:20px; left:25px;">Hasil Akhir</label>
            <div id="res-main" class="math-display">$$ \text{Menunggu Input...} $$</div>
        </div>

        <div id="visual-panel" class="card-visual">
            <div class="visual-title">
                <span><span style="font-size:1.2em; margin-right:5px;">📐</span> Visualisasi Geometris</span>
                <span class="badge">GENERATED</span>
            </div>
            <div id="visual-container" style="width:100%; text-align:center;">
                </div>
        </div>

        <div class="card-explanation">
            <h3 style="margin-top:0; color:var(--accent-light); display:flex; align-items:center; gap:8px;">
                <span>💡</span> Langkah Pengerjaan
            </h3>
            <div id="res-explain">
                <div class="step-item">Silakan pilih modul dan masukkan nilai.</div>
            </div>
        </div>
    </div>
</div>

<script>
    let currentModule = 'geo';

    function switchTab(mod) {
        currentModule = mod;
        document.getElementById('tab-geo').classList.toggle('active', mod === 'geo');
        document.getElementById('tab-trig').classList.toggle('active', mod === 'trig');
        document.getElementById('form-geo').classList.toggle('hidden', mod !== 'geo');
        document.getElementById('form-trig').classList.toggle('hidden', mod !== 'trig');
        if(mod === 'geo') updateGeoForm(); else updateTrigForm();
    }
    
    function updateGeoForm() {
        const op = document.getElementById('geo-op').value;
        const ids = ['geo-vector', 'geo-refleksi', 'geo-rotasi', 'geo-dilatasi', 'geo-invers'];
        ids.forEach(id => document.getElementById(id).classList.add('hidden'));
        
        if(op.includes('translasi')) document.getElementById('geo-vector').classList.remove('hidden');
        else if(op === 'refleksi') document.getElementById('geo-refleksi').classList.remove('hidden');
        else if(op === 'rotasi') document.getElementById('geo-rotasi').classList.remove('hidden');
        else if(op === 'dilatasi') document.getElementById('geo-dilatasi').classList.remove('hidden');
        else if(op === 'invers') document.getElementById('geo-invers').classList.remove('hidden');
    }
    
    function updateTrigForm() {
        const op = document.getElementById('trig-op').value;
        ['trig-sinus', 'trig-sinus-ambigu', 'trig-cosinus', 'trig-luas'].forEach(id => 
            document.getElementById(id).classList.add('hidden'));
        
        if(op === 'aturan_sinus') document.getElementById('trig-sinus').classList.remove('hidden');
        else if(op === 'aturan_sinus_ambigu') document.getElementById('trig-sinus-ambigu').classList.remove('hidden');
        else if(op === 'aturan_cosinus') {
            document.getElementById('trig-cosinus').classList.remove('hidden');
            updateCosForm();
        } else if(op === 'luas_segitiga') document.getElementById('trig-luas').classList.remove('hidden');
    }
    
    function updateCosForm() {
        const type = document.getElementById('trig-cos-type').value;
        document.getElementById('trig-cos-sisi').classList.toggle('hidden', type !== 'cari_sisi');
        document.getElementById('trig-cos-sudut').classList.toggle('hidden', type !== 'cari_sudut');
    }
    
    function calculate() {
        const btn = document.querySelector('.btn-calc');
        btn.innerHTML = "⏳ MEMPROSES...";
        
        // Reset Visual Panel
        document.getElementById('visual-panel').style.display = 'none';
        document.getElementById('visual-container').innerHTML = '';
        
        let payload = {};
        
        // GEOMETRY PAYLOAD
        if (currentModule === 'geo') {
            const op = document.getElementById('geo-op').value;
            payload = { module: 'geo', operation: op, px: document.getElementById('geo-px').value, py: document.getElementById('geo-py').value };
            
            if (op.includes('translasi')) { payload.tx = document.getElementById('geo-tx').value; payload.ty = document.getElementById('geo-ty').value; }
            else if (op === 'refleksi') payload.mode = document.getElementById('geo-mode').value;
            else if (op === 'rotasi') { payload.angle = document.getElementById('geo-angle').value; payload.cx = document.getElementById('geo-cx').value; payload.cy = document.getElementById('geo-cy').value; }
            else if (op === 'dilatasi') { payload.factor = document.getElementById('geo-factor').value; payload.dcx = document.getElementById('geo-dcx').value; payload.dcy = document.getElementById('geo-dcy').value; }
            else if (op === 'invers') { payload.inv_type = document.getElementById('geo-inv-type').value; payload.param = document.getElementById('geo-param').value; }
        
        // TRIGONOMETRY PAYLOAD
        } else {
            const op = document.getElementById('trig-op').value;
            payload = { module: 'trig', operation: op };
            
            if (op === 'aturan_sinus') {
                payload.b = document.getElementById('trig-b').value;
                payload.A = document.getElementById('trig-A').value;
                payload.B = document.getElementById('trig-B').value;
            } else if (op === 'aturan_sinus_ambigu') {
                payload.a = document.getElementById('trig-a-ambi').value;
                payload.b = document.getElementById('trig-b-ambi').value;
                payload.A = document.getElementById('trig-A-ambi').value;
            } else if (op === 'aturan_cosinus') {
                const type = document.getElementById('trig-cos-type').value;
                if (type === 'cari_sisi') {
                    payload.cari = 'sisi';
                    payload.a = document.getElementById('trig-a-cos').value;
                    payload.b = document.getElementById('trig-b-cos').value;
                    payload.C = document.getElementById('trig-C-angle').value;
                } else {
                    payload.cari = 'sudut';
                    payload.a = document.getElementById('trig-a-cos2').value;
                    payload.b = document.getElementById('trig-b-cos2').value;
                    payload.c = document.getElementById('trig-c-cos2').value;
                }
            } else if (op === 'luas_segitiga') {
                payload.a = document.getElementById('trig-a-luas').value;
                payload.b = document.getElementById('trig-b-luas').value;
                payload.C = document.getElementById('trig-C-luas').value;
            }
        }
        
        fetch('/compute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            // Render Result
            if (data.error) {
                document.getElementById('res-main').innerHTML = `<div style="color:${data.color || '#ef4444'}">${data.error}</div>`;
            } else {
                let mainHTML = '';
                if(Array.isArray(data.result)) data.result.forEach(r => mainHTML += `<div>${r}</div>`);
                else mainHTML = `<div>${data.result}</div>`;
                
                if(data.matrix) mainHTML += `<div style="margin-top:10px; font-size:0.8em; color:var(--text-dim)">${data.matrix}</div>`;
                document.getElementById('res-main').innerHTML = mainHTML;
                
                // RENDER STEPS
                let stepsHTML = '';
                if(data.status) stepsHTML += `<div class="status-box ${data.status_class}">${data.status}</div>`;
                if(data.steps) data.steps.forEach(s => stepsHTML += `<div class="step-item"><b>${s.title}</b> ${s.desc}</div>`);
                document.getElementById('res-explain').innerHTML = stepsHTML;

                // RENDER VISUALIZATION (NEW)
                if(data.images && data.images.length > 0) {
                    const visPanel = document.getElementById('visual-panel');
                    const visContainer = document.getElementById('visual-container');
                    visPanel.style.display = 'flex';
                    
                    data.images.forEach(imgData => {
                        visContainer.innerHTML += `
                            <img src="data:image/png;base64,${imgData}" class="visual-img" style="margin-bottom:15px;">
                        `;
                    });
                } else if (data.image) {
                     const visPanel = document.getElementById('visual-panel');
                     visPanel.style.display = 'flex';
                     document.getElementById('visual-container').innerHTML = `
                        <img src="data:image/png;base64,${data.image}" class="visual-img">
                     `;
                }
            }
            MathJax.typesetPromise();
            btn.innerHTML = "HITUNG SEKARANG";
        })
        .catch(err => {
            btn.innerHTML = "HITUNG SEKARANG";
            alert("Error: " + err);
        });
    }
    
    document.addEventListener('DOMContentLoaded', () => { updateGeoForm(); MathJax.typesetPromise(); });
</script>

</body>
</html>
"""

# =======================
# ROUTES
# =======================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/compute", methods=["POST"])
def compute():
    data = request.json
    mod = data.get('module')
    op = data.get('operation')
    
    try:
        if mod == 'geo':
            # Parsing input dasar
            px = sp.sympify(data.get('px', '0'))
            py = sp.sympify(data.get('py', '0'))
            
            if op == 'translasi':
                tx = sp.sympify(data.get('tx', '0'))
                ty = sp.sympify(data.get('ty', '0'))
                res, step_list = GeoEngine.translasi((px, py), (tx, ty))
                steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                return jsonify({
                    "result": f"P'({fnum(res[0])}, {fnum(res[1])})",
                    "steps": steps,
                    "status": "✓ Translasi selesai",
                    "status_class": "success"
                })
                
            elif op == 'translasi_homogen':
                tx = sp.sympify(data.get('tx', '0'))
                ty = sp.sympify(data.get('ty', '0'))
                res, step_list, mat = GeoEngine.translasi_homogen((px, py), (tx, ty))
                steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                return jsonify({
                    "result": f"P'({fnum(res[0])}, {fnum(res[1])})",
                    "matrix": f"Matriks: {sp.latex(mat)}",
                    "steps": steps,
                    "status": "✓ Translasi Homogen selesai",
                    "status_class": "success"
                })

            elif op == 'refleksi':
                mode = data.get('mode', 'x')
                res, step_list, mat = GeoEngine.refleksi((px, py), mode)
                steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                return jsonify({
                    "result": f"P'({fnum(res[0])}, {fnum(res[1])})",
                    "matrix": f"Matriks: {sp.latex(mat)}",
                    "steps": steps,
                    "status": "✓ Refleksi selesai",
                    "status_class": "success"
                })

            elif op == 'rotasi':
                angle = sp.sympify(data.get('angle', '90'))
                cx = sp.sympify(data.get('cx', '0'))
                cy = sp.sympify(data.get('cy', '0'))
                res, step_list, mat = GeoEngine.rotasi((px, py), angle, (cx, cy))
                steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                return jsonify({
                    "result": f"P'({fnum(res[0])}, {fnum(res[1])})",
                    "matrix": f"Matriks: {sp.latex(mat)}",
                    "steps": steps,
                    "status": "✓ Rotasi selesai",
                    "status_class": "success"
                })

            elif op == 'dilatasi':
                factor = sp.sympify(data.get('factor', '2'))
                dcx = sp.sympify(data.get('dcx', '0'))
                dcy = sp.sympify(data.get('dcy', '0'))
                res, step_list, mat = GeoEngine.dilatasi((px, py), factor, (dcx, dcy))
                steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                return jsonify({
                    "result": f"P'({fnum(res[0])}, {fnum(res[1])})",
                    "matrix": f"Matriks: {sp.latex(mat)}",
                    "steps": steps,
                    "status": "✓ Dilatasi selesai",
                    "status_class": "success"
                })

            elif op == 'invers':
                inv_type = data.get('inv_type', 'rot')
                param = sp.sympify(data.get('param', '90'))
                mat_inv, step_list = GeoEngine.invers_transformasi(inv_type, param)
                steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                if mat_inv is not None:
                    return jsonify({
                        "result": "Invers Matriks Ditemukan",
                        "matrix": f"$$M^{{-1}} = {sp.latex(mat_inv)}$$",
                        "steps": steps,
                        "status": "✓ Perhitungan sukses",
                        "status_class": "success"
                    })
                return jsonify({"error": "Matriks singular", "steps": steps})

        elif mod == 'trig':
            if op == 'aturan_sinus':
                b = sp.sympify(data.get('b', '5'))
                A = sp.sympify(data.get('A', '30'))
                B = sp.sympify(data.get('B', '45'))
                
                rad_A = to_rad(A)
                rad_B = to_rad(B)
                a = (b * sp.sin(rad_A)) / sp.sin(rad_B)
                
                # Hitung C untuk visualisasi
                C_angle = 180 - float(A) - float(B)
                c_side = (b * sp.sin(to_rad(C_angle))) / sp.sin(rad_B)
                img = Plotter.create_triangle_image(a, b, c_side, float(A), float(B), C_angle)
                
                steps = [
                    {"title": "Rumus", "desc": "a/sin A = b/sin B"},
                    {"title": "Substitusi", "desc": f"a = {b} × sin({A}) / sin({B})"},
                    {"title": "Hasil", "desc": f"a = {fnum(a)}"}
                ]
                return jsonify({
                    "result": f"Sisi a = {fnum(a)}",
                    "steps": steps,
                    "image": img,
                    "status": "✓ Sisi a ditemukan",
                    "status_class": "success"
                })
                
            elif op == 'aturan_sinus_ambigu':
                a = sp.sympify(data.get('a', '5'))
                b = sp.sympify(data.get('b', '7'))
                A = sp.sympify(data.get('A', '30'))
                res_list, step_list, status, images = TrigEngine.aturan_sinus_ambigu(a, b, A)
                steps = [{"title": f"Analisis", "desc": s} for s in step_list]
                
                result_txt = []
                if not res_list: result_txt.append("Tidak ada solusi")
                else: 
                    for i, ang in enumerate(res_list): result_txt.append(f"Sudut B{i+1} = {fnum(ang)}°")

                return jsonify({
                    "result": result_txt,
                    "steps": steps,
                    "status": status,
                    "status_class": "warning" if "Ambigu" in status else "success",
                    "images": images
                })
                
            elif op == 'aturan_cosinus':
                cari = data.get('cari', 'sisi')
                if cari == 'sisi':
                    a = sp.sympify(data.get('a', '5'))
                    b = sp.sympify(data.get('b', '6'))
                    C = sp.sympify(data.get('C', '60'))
                    res, step_list, img = TrigEngine.aturan_cosinus(a=a, b=b, angle_C=C)
                    steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                    return jsonify({
                        "result": f"Sisi c = {fnum(res)}",
                        "steps": steps,
                        "image": img,
                        "status": "✓ Sisi c ditemukan",
                        "status_class": "success"
                    })
                else:
                    a = sp.sympify(data.get('a', '5'))
                    b = sp.sympify(data.get('b', '6'))
                    c = sp.sympify(data.get('c', '7'))
                    res, step_list, img = TrigEngine.aturan_cosinus(a=a, b=b, c=c)
                    steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                    return jsonify({
                        "result": f"Sudut C = {fnum(res)}°",
                        "steps": steps,
                        "image": img,
                        "status": "✓ Sudut C ditemukan",
                        "status_class": "success"
                    })

            elif op == 'luas_segitiga':
                a = sp.sympify(data.get('a', '5'))
                b = sp.sympify(data.get('b', '6'))
                C = sp.sympify(data.get('C', '30'))
                res, step_list, img = TrigEngine.luas_segitiga(a, b, C)
                steps = [{"title": f"Langkah {i+1}", "desc": s} for i, s in enumerate(step_list)]
                return jsonify({
                    "result": f"Luas = {fnum(res)} satuan²",
                    "steps": steps,
                    "image": img,
                    "status": "✓ Luas segitiga dihitung",
                    "status_class": "success"
                })

    except Exception as e:
        return jsonify({"error": f"Input Error: {str(e)}", "color": "#ef4444"})

    return jsonify({"error": "Operasi tidak valid"})

# =======================
# AUTO START
# =======================
def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=False, port=5000)
