"""
Scholarship Allocation Prediction GUI
======================================
A GUI application that predicts whether a student will be
Awarded or Rejected for a scholarship based on their details.

Uses a Decision Tree model trained on the dataset_combined.xlsx file.
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os
import threading

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_combined.xlsx")

# Colour palette (matching the notebook)
BG_MAIN       = "#f0f4f8"
BG_PANEL      = "#ffffff"
BG_HEADER     = "#1a3a5c"
FG_HEADER     = "#ffffff"
COLOR_AWARDED  = "#1baf7a"   # green
COLOR_REJECTED = "#eb6834"   # orange
COLOR_BTN      = "#2a78d6"   # blue
COLOR_BTN_FG   = "#ffffff"
COLOR_ACCENT   = "#2a78d6"

# Scholarship schemes (shortened names matching the notebook)
SCHEMES = [
    "INSPIRE-SHE",
    "Abdul Kalam Fellowship",
    "AAI Sports",
    "Glow & Lovely",
    "NF Disabilities",
    "ONGC Sports",
    "Pragati (AICTE)",
    "Dr. Ambedkar",
    "Indira Gandhi",
    "National Overseas",
]

EDUCATION_OPTIONS  = ["Undergraduate", "Postgraduate", "Doctorate"]
GENDER_OPTIONS     = ["Male", "Female"]
COMMUNITY_OPTIONS  = ["General", "OBC", "SC/ST", "Minority"]
RELIGION_OPTIONS   = ["Hindu", "Muslim", "Christian", "Others"]
EXSERVICE_OPTIONS  = ["Yes", "No"]
DISABILITY_OPTIONS = ["Yes", "No"]
SPORTS_OPTIONS     = ["Yes", "No"]
PERCENTAGE_OPTIONS = ["90-100", "80-90", "70-80", "60-70"]
INCOME_OPTIONS     = ["Upto 1.5L", "1.5L to 3L", "3L to 6L", "Above 6L"]
INDIA_OPTIONS      = ["In", "Out"]   # In = India, Out = Overseas


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
class ScholarshipModel:
    """Loads dataset, trains Decision Tree, exposes predict()."""

    SHORT = {
        "INSPIRE Scholarship 2022-23 ? Scholarship for Higher Education (SHE)": "INSPIRE-SHE",
        "Abdul Kalam Technology Innovation National Fellowship": "Abdul Kalam Fellowship",
        "AAI Sports Scholarship Scheme in India 2022-23": "AAI Sports",
        "Glow and lovely Career Foundation Scholarship": "Glow & Lovely",
        "National Fellowship for Persons with Disabilities": "NF Disabilities",
        "ONGC Sports Scholarship Scheme 2022-23": "ONGC Sports",
        "Pragati Scholarship ? AICTE-Scholarship Scheme to Girl Child": "Pragati (AICTE)",
        "Dr. Ambedkar post matric Scholarship": "Dr. Ambedkar",
        "Indira Gandhi Scholarship for Single Girl Child UGC Scholarship for PG Programmes": "Indira Gandhi",
        "National Overseas Scholarship Scheme 2021-22": "National Overseas",
    }

    COLS = ["Scheme", "Education", "Gender", "Community", "Religion",
            "ExService", "Disability", "Sports", "Percentage", "Income", "India"]

    def __init__(self):
        self.model = None
        self.encoders = {}
        self.trained = False
        self.error = None

    def load_and_train(self):
        try:
            df = pd.read_excel(DATASET_PATH)
            df.columns = df.columns.str.strip()

            # Rename
            df = df.rename(columns={
                "Name": "Scheme",
                "Education Qualification": "Education",
                "Exservice-men": "ExService",
                "Annual-Percentage": "Percentage",
            })

            # Fix spelling
            df["Education"] = df["Education"].replace({"Doctrate": "Doctorate"})
            df["Religion"]  = df["Religion"].replace({"Chirstian": "Christian"})

            # Shorten scheme names
            if "Scheme" in df.columns:
                df["Scheme"] = df["Scheme"].replace(self.SHORT)

            # Encode categoricals
            X = df[self.COLS].copy()
            y = df["Outcome"].values

            for col in self.COLS:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le

            # Train Decision Tree
            self.model = DecisionTreeClassifier(random_state=42)
            self.model.fit(X, y)
            self.trained = True
        except Exception as e:
            self.error = str(e)

    def predict(self, inputs: dict):
        """inputs: dict with keys matching COLS."""
        row = {}
        for col in self.COLS:
            val = inputs[col]
            le  = self.encoders[col]
            if val not in le.classes_:
                # fallback: use closest or first
                val = le.classes_[0]
            row[col] = le.transform([val])[0]

        X = pd.DataFrame([row])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return pred, proba


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────
class ScholarshipApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎓 Scholarship Allocation Predictor")
        self.geometry("820x720")
        self.configure(bg=BG_MAIN)
        self.resizable(True, True)

        self.model = ScholarshipModel()
        self._build_ui()
        self._start_model_loading()

    # ── UI builder ──────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────
        hdr = tk.Frame(self, bg=BG_HEADER, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎓  Scholarship Allocation Predictor",
                 font=("Helvetica", 20, "bold"),
                 bg=BG_HEADER, fg=FG_HEADER).pack()
        tk.Label(hdr, text="Data Mining & Data Warehousing  |  BTCO14505  |  Open Ended Project",
                 font=("Helvetica", 10),
                 bg=BG_HEADER, fg="#a8c8e8").pack()

        # ── Status bar ──────────────────────────
        self.status_var = tk.StringVar(value="⏳  Loading dataset and training model…")
        self.status_lbl = tk.Label(self, textvariable=self.status_var,
                                   font=("Helvetica", 9, "italic"),
                                   bg=BG_MAIN, fg="#666666", anchor="w", padx=10)
        self.status_lbl.pack(fill="x", pady=(4, 0))

        # ── Scrollable content area ──────────────
        canvas = tk.Canvas(self, bg=BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=BG_MAIN)

        self.scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # mouse-wheel binding
        self.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ── Form ────────────────────────────────
        self._build_form(self.scroll_frame)

        # ── Result panel ────────────────────────
        self._build_result_panel(self.scroll_frame)

    def _section_label(self, parent, text):
        frm = tk.Frame(parent, bg=BG_MAIN)
        frm.pack(fill="x", padx=10, pady=(14, 2))
        tk.Label(frm, text=text, font=("Helvetica", 12, "bold"),
                 bg=BG_MAIN, fg=BG_HEADER).pack(anchor="w")
        sep = tk.Frame(frm, bg=COLOR_ACCENT, height=2)
        sep.pack(fill="x", pady=(2, 0))

    def _combo_row(self, parent, label, options, default=None):
        """Returns a StringVar tied to a combo box row."""
        row = tk.Frame(parent, bg=BG_PANEL, pady=6, padx=12)
        row.pack(fill="x", padx=10, pady=2)

        tk.Label(row, text=label, width=22, anchor="w",
                 font=("Helvetica", 10), bg=BG_PANEL).pack(side="left")

        var = tk.StringVar(value=default or options[0])
        cb  = ttk.Combobox(row, textvariable=var, values=options,
                            state="readonly", width=28,
                            font=("Helvetica", 10))
        cb.pack(side="left", padx=(6, 0))
        return var

    def _build_form(self, parent):
        # Panel
        panel = tk.Frame(parent, bg=BG_PANEL, bd=1, relief="flat")
        panel.pack(fill="x", padx=10, pady=(0, 4))

        self._section_label(parent, "📋  Student Details")

        inner = tk.Frame(parent, bg=BG_PANEL)
        inner.pack(fill="x")

        self.v_scheme     = self._combo_row(inner, "Scholarship Scheme", SCHEMES)
        self.v_education  = self._combo_row(inner, "Education Level",    EDUCATION_OPTIONS)
        self.v_gender     = self._combo_row(inner, "Gender",             GENDER_OPTIONS)
        self.v_community  = self._combo_row(inner, "Community",          COMMUNITY_OPTIONS)
        self.v_religion   = self._combo_row(inner, "Religion",           RELIGION_OPTIONS)
        self.v_exservice  = self._combo_row(inner, "Ex-Serviceman",      EXSERVICE_OPTIONS, "No")
        self.v_disability = self._combo_row(inner, "Disability",         DISABILITY_OPTIONS, "No")
        self.v_sports     = self._combo_row(inner, "Sports Achievement", SPORTS_OPTIONS, "No")
        self.v_percentage = self._combo_row(inner, "Annual Percentage",  PERCENTAGE_OPTIONS)
        self.v_income     = self._combo_row(inner, "Annual Income",      INCOME_OPTIONS)
        self.v_india      = self._combo_row(inner, "Studying In / Out",  INDIA_OPTIONS)

        # Predict button
        btn_frame = tk.Frame(parent, bg=BG_MAIN, pady=12)
        btn_frame.pack()
        self.predict_btn = tk.Button(
            btn_frame, text="🔍  Predict Scholarship Outcome",
            command=self._on_predict,
            font=("Helvetica", 13, "bold"),
            bg=COLOR_BTN, fg=COLOR_BTN_FG,
            activebackground="#1a5fa0", activeforeground="white",
            padx=20, pady=10, bd=0, cursor="hand2",
            state="disabled"
        )
        self.predict_btn.pack()

    def _build_result_panel(self, parent):
        self._section_label(parent, "📊  Prediction Result")

        self.result_frame = tk.Frame(parent, bg=BG_PANEL, bd=2, relief="groove",
                                     padx=20, pady=20)
        self.result_frame.pack(fill="x", padx=10, pady=(0, 16))

        self.result_icon = tk.Label(self.result_frame, text="",
                                    font=("Helvetica", 48), bg=BG_PANEL)
        self.result_icon.pack()

        self.result_title = tk.Label(self.result_frame, text="Awaiting prediction…",
                                     font=("Helvetica", 18, "bold"),
                                     bg=BG_PANEL, fg="#999999")
        self.result_title.pack(pady=(4, 0))

        self.result_sub = tk.Label(self.result_frame, text="",
                                   font=("Helvetica", 11),
                                   bg=BG_PANEL, fg="#555555", wraplength=560)
        self.result_sub.pack(pady=(4, 8))

        # Confidence bars
        bar_frm = tk.Frame(self.result_frame, bg=BG_PANEL)
        bar_frm.pack(fill="x", padx=20)

        tk.Label(bar_frm, text="Awarded probability:", width=24, anchor="w",
                 font=("Helvetica", 10), bg=BG_PANEL).grid(row=0, column=0, sticky="w")
        self.bar_awarded = ttk.Progressbar(bar_frm, length=300, maximum=100)
        self.bar_awarded.grid(row=0, column=1, padx=(6, 6))
        self.lbl_awarded = tk.Label(bar_frm, text="—",
                                    font=("Helvetica", 10, "bold"),
                                    bg=BG_PANEL, fg=COLOR_AWARDED, width=7)
        self.lbl_awarded.grid(row=0, column=2)

        tk.Label(bar_frm, text="Rejected probability:", width=24, anchor="w",
                 font=("Helvetica", 10), bg=BG_PANEL).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.bar_rejected = ttk.Progressbar(bar_frm, length=300, maximum=100)
        self.bar_rejected.grid(row=1, column=1, padx=(6, 6), pady=(4, 0))
        self.lbl_rejected = tk.Label(bar_frm, text="—",
                                     font=("Helvetica", 10, "bold"),
                                     bg=BG_PANEL, fg=COLOR_REJECTED, width=7)
        self.lbl_rejected.grid(row=1, column=2, pady=(4, 0))

    # ── Model loading ───────────────────────────
    def _start_model_loading(self):
        t = threading.Thread(target=self._load_model_thread, daemon=True)
        t.start()

    def _load_model_thread(self):
        self.model.load_and_train()
        self.after(0, self._on_model_ready)

    def _on_model_ready(self):
        if self.model.error:
            self.status_var.set(f"  Error: {self.model.error}")
            messagebox.showerror("Dataset Error",
                f"Could not load dataset:\n{self.model.error}\n\n"
                f"Expected path:\n{DATASET_PATH}")
        else:
            self.status_var.set("  Model trained and ready! Fill in the details and click Predict.")
            self.status_lbl.config(fg="#1baf7a")
            self.predict_btn.config(state="normal")

    # ── Prediction ──────────────────────────────
    def _on_predict(self):
        if not self.model.trained:
            messagebox.showwarning("Not ready", "Model is still loading. Please wait.")
            return

        inputs = {
            "Scheme":     self.v_scheme.get(),
            "Education":  self.v_education.get(),
            "Gender":     self.v_gender.get(),
            "Community":  self.v_community.get(),
            "Religion":   self.v_religion.get(),
            "ExService":  self.v_exservice.get(),
            "Disability": self.v_disability.get(),
            "Sports":     self.v_sports.get(),
            "Percentage": self.v_percentage.get(),
            "Income":     self.v_income.get(),
            "India":      self.v_india.get(),
        }

        pred, proba = self.model.predict(inputs)

        # proba: index 0 = Rejected, index 1 = Awarded
        p_awarded  = proba[1] * 100
        p_rejected = proba[0] * 100

        self.bar_awarded["value"]  = p_awarded
        self.bar_rejected["value"] = p_rejected
        self.lbl_awarded.config(text=f"{p_awarded:.1f}%")
        self.lbl_rejected.config(text=f"{p_rejected:.1f}%")

        if pred == 1:
            self.result_icon.config(text="🏆", fg=COLOR_AWARDED)
            self.result_title.config(
                text="✅  Scholarship AWARDED!",
                fg=COLOR_AWARDED)
            self.result_sub.config(
                text=f"Congratulations! Based on the provided details, "
                     f"this student is likely to be awarded the scholarship "
                     f"({self.v_scheme.get()}) with {p_awarded:.1f}% confidence.",
                fg="#2d6a4f")
            self.result_frame.config(bg="#f0fff8")
            for w in self.result_frame.winfo_children():
                if isinstance(w, tk.Label):
                    w.config(bg="#f0fff8")
        else:
            self.result_icon.config(text="❌", fg=COLOR_REJECTED)
            self.result_title.config(
                text="❌  Scholarship REJECTED",
                fg=COLOR_REJECTED)
            self.result_sub.config(
                text=f"Based on the provided details, this student is likely to be "
                     f"rejected for the scholarship ({self.v_scheme.get()}) "
                     f"with {p_rejected:.1f}% confidence. "
                     f"Consider reviewing the eligibility criteria.",
                fg="#7d3c1a")
            self.result_frame.config(bg="#fff5f0")
            for w in self.result_frame.winfo_children():
                if isinstance(w, tk.Label):
                    w.config(bg="#fff5f0")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ScholarshipApp()
    app.mainloop()
