"""
Barcode & Invoice-Number PDF Renamer
--------------------------------------
Two tabs:

1) Barcode Rename
   Scans a folder of scanned PDFs, reads the barcode printed on the page
   (Code 39 / Code 128 / QR / Data Matrix all supported), and renames
   (or copies) each PDF to the barcode's decoded value.

2) Invoice Number (OCR)
   For scanned Iranian sales-invoice PDFs with a "7/ XXXXX" style number
   printed near the top of the page. Crops the number field, runs OCR,
   and renames (or copies) each PDF to "7XXXXX.pdf". Because the exact
   position/orientation of that field can vary between scan batches, this
   tab includes a live Preview so you can calibrate the crop box and
   rotation against your own files before running the full batch.

Both tabs share: dark mode, a "failed" subfolder for unreadable files,
and a "Move renamed files..." button to relocate successful results.

Run with:  python BarcodeRenamer.py
Requires:
  pip install pymupdf pyzbar zxing-cpp pytesseract pillow numpy ttkbootstrap
Also needs, installed separately (not pip-installable):
  - Tesseract OCR engine (Windows): https://github.com/UB-Mannheim/tesseract/wiki
    During install, add the Persian ("fas") language under "Additional language data".
    If tesseract.exe isn't on PATH, set the path in the Invoice Number tab.
"""

import os
import sys
import re
import shutil
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import fitz  # PyMuPDF
import zxingcpp
from PIL import Image, ImageTk

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAVE_PYZBAR = True
except Exception:
    HAVE_PYZBAR = False

try:
    import pytesseract
    HAVE_TESSERACT = True
except Exception:
    HAVE_TESSERACT = False

import ttkbootstrap as tb
from ttkbootstrap.constants import *

try:
    from invoice_financial import (
        DigitModel, process_folder, summarize, export_excel,
    )
    HAVE_INVOICE_FINANCIAL = True
except Exception:
    HAVE_INVOICE_FINANCIAL = False


def get_app_path(relative_path=""):
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def configure_bundled_tesseract():
    tesseract_exe = get_app_path(
        os.path.join("tesseract", "tesseract.exe")
    )

    tessdata_dir = get_app_path(
        os.path.join("tesseract", "tessdata")
    )

    if os.path.isfile(tesseract_exe):
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe

        if os.path.isdir(tessdata_dir):
            os.environ["TESSDATA_PREFIX"] = tessdata_dir

        return tesseract_exe

    return None


if HAVE_TESSERACT:
    BUNDLED_TESSERACT_PATH = configure_bundled_tesseract()
else:
    BUNDLED_TESSERACT_PATH = None
# ---------------------------------------------------------------- constants

BARCODE_RENDER_DPI = 250
MAX_PAGES_TO_TRY = 2          # look at first N pages if barcode isn't on page 1
INVALID_CHARS = r'[\\/:*?"<>|]'
MAX_NAME_LEN = 150

LIGHT_THEME = "flatly"
DARK_THEME = "darkly"

LOG_COLORS = {
    "light": {
        "bg": "#ffffff", "fg": "#212529",
        "ok": "#198754", "err": "#dc3545", "warn": "#fd7e14",
        "done": "#0d6efd", "info": "#6c757d",
    },
    "dark": {
        "bg": "#1a1d21", "fg": "#e9ecef",
        "ok": "#57e389", "err": "#ff6b6b", "warn": "#ffb454",
        "done": "#66b2ff", "info": "#9aa1a9",
    },
}

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def normalize_digits(s):
    """Convert Persian/Arabic-Indic digits in a string to plain ASCII digits."""
    trans = {}
    for i, ch in enumerate(PERSIAN_DIGITS):
        trans[ch] = str(i)
    for i, ch in enumerate(ARABIC_DIGITS):
        trans[ch] = str(i)
    return "".join(trans.get(c, c) for c in s)


def sanitize_filename(text):
    text = text.strip()
    text = re.sub(INVALID_CHARS, "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "UNKNOWN"
    return text[:MAX_NAME_LEN]


def unique_destination(folder, base_name, ext):
    candidate = os.path.join(folder, base_name + ext)
    if not os.path.exists(candidate):
        return candidate
    i = 2
    while True:
        candidate = os.path.join(folder, f"{base_name} ({i}){ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def list_pdfs_in_folder(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    )


# ============================================================== BARCODE TAB

def decode_barcode_from_page(page):
    """Render a PDF page and try to decode a barcode from it. Returns (text, format) or (None, None)."""
    pix = page.get_pixmap(dpi=BARCODE_RENDER_DPI, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    arr = np.array(img)

    results = zxingcpp.read_barcodes(arr)
    if results:
        return results[0].text, str(results[0].format)

    if HAVE_PYZBAR:
        pz = pyzbar_decode(img)
        if pz:
            val = pz[0].data.decode("utf-8", errors="ignore")
            return val, str(pz[0].type)

    return None, None


class BarcodeRenameWorker(threading.Thread):
    def __init__(self, source_folder, mode, output_folder, msg_queue, stop_event):
        super().__init__(daemon=True)
        self.source_folder = source_folder
        self.mode = mode
        self.output_folder = output_folder
        self.q = msg_queue
        self.stop_event = stop_event

    def log(self, text, tag="info"):
        self.q.put(("log", (text, tag)))

    def run(self):
        try:
            files = [os.path.basename(p) for p in list_pdfs_in_folder(self.source_folder)]
            total = len(files)
            self.log(f"Found {total} PDF file(s) in source folder.")
            self.q.put(("progress_max", total))

            if self.mode == "copy":
                dest_root = self.output_folder
                failed_root = os.path.join(self.output_folder, "failed")
            else:
                dest_root = self.source_folder
                failed_root = os.path.join(self.source_folder, "failed")

            os.makedirs(dest_root, exist_ok=True)

            renamed, failed = 0, 0

            for idx, fname in enumerate(files, start=1):
                if self.stop_event.is_set():
                    self.log("Cancelled by user.", "warn")
                    break

                src_path = os.path.join(self.source_folder, fname)
                barcode_text = None
                barcode_fmt = None

                try:
                    doc = fitz.open(src_path)
                    pages_to_try = min(len(doc), MAX_PAGES_TO_TRY)
                    for p in range(pages_to_try):
                        barcode_text, barcode_fmt = decode_barcode_from_page(doc[p])
                        if barcode_text:
                            break
                    doc.close()
                except Exception as e:
                    self.log(f"\u2717 ERROR opening {fname}: {e}", "err")

                try:
                    if barcode_text:
                        new_base = sanitize_filename(barcode_text)
                        dest_path = unique_destination(dest_root, new_base, ".pdf")

                        if self.mode == "copy":
                            shutil.copy2(src_path, dest_path)
                        else:
                            os.rename(src_path, dest_path)

                        renamed += 1
                        self.log(
                            f"\u2713 {fname}  \u2192  {os.path.basename(dest_path)}  "
                            f"[{barcode_fmt}]", "ok"
                        )
                        self.q.put(("success_path", dest_path))
                    else:
                        os.makedirs(failed_root, exist_ok=True)
                        dest_path = unique_destination(failed_root, os.path.splitext(fname)[0], ".pdf")

                        if self.mode == "copy":
                            shutil.copy2(src_path, dest_path)
                        else:
                            os.rename(src_path, dest_path)

                        failed += 1
                        self.log(f"\u2717 No barcode found: {fname}  \u2192  moved to failed/", "warn")

                except Exception as e:
                    failed += 1
                    self.log(f"\u2717 FAILED handling {fname}: {e}", "err")

                self.q.put(("progress", idx))
                self.q.put(("counts", (renamed, failed, total)))

            self.log(f"Done. Renamed: {renamed}  Failed: {failed}  Total: {total}", "done")

        except Exception as e:
            self.log(f"FATAL ERROR: {e}", "err")
        finally:
            self.q.put(("finished", None))


class BarcodeTab(tb.Frame):
    """Original barcode-based renamer, as its own tab."""

    def __init__(self, master, get_dark_mode):
        super().__init__(master, padding=16)
        self.get_dark_mode = get_dark_mode
        self.msg_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.mode_var = tk.StringVar(value="copy")
        self.successful_paths = []

        self._build_ui()
        self._on_mode_change()
        self.apply_log_colors()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        config_card = tb.Labelframe(self, text="  Configuration  ", padding=16)
        config_card.pack(fill=X, pady=(0, 14))
        config_card.columnconfigure(0, weight=1)

        tb.Label(config_card, text="Source folder (contains the PDFs)", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        src_row = tb.Frame(config_card)
        src_row.grid(row=1, column=0, sticky="ew", pady=(2, 12))
        src_row.columnconfigure(0, weight=1)

        self.source_var = tk.StringVar()
        tb.Entry(src_row, textvariable=self.source_var).grid(row=0, column=0, sticky="ew")
        tb.Button(
            src_row, text="Browse...", bootstyle=(SECONDARY, OUTLINE),
            command=self._browse_source
        ).grid(row=0, column=1, padx=(8, 0))

        tb.Label(config_card, text="What should happen to renamed files?", font=("Segoe UI", 9, "bold")).grid(
            row=2, column=0, sticky="w"
        )
        mode_row = tb.Frame(config_card)
        mode_row.grid(row=3, column=0, sticky="ew", pady=(4, 12))

        tb.Radiobutton(
            mode_row, text="Rename in place (source folder)", value="in_place",
            variable=self.mode_var, bootstyle="toolbutton", command=self._on_mode_change
        ).pack(side=LEFT, padx=(0, 8))
        tb.Radiobutton(
            mode_row, text="Copy renamed files to a new folder", value="copy",
            variable=self.mode_var, bootstyle="toolbutton", command=self._on_mode_change
        ).pack(side=LEFT)

        self.output_label = tb.Label(config_card, text="Output folder", font=("Segoe UI", 9, "bold"))
        self.output_label.grid(row=4, column=0, sticky="w")
        out_row = tb.Frame(config_card)
        out_row.grid(row=5, column=0, sticky="ew", pady=(2, 4))
        out_row.columnconfigure(0, weight=1)

        self.output_var = tk.StringVar()
        self.output_entry = tb.Entry(out_row, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        self.output_browse_btn = tb.Button(
            out_row, text="Browse...", bootstyle=(SECONDARY, OUTLINE),
            command=self._browse_output
        )
        self.output_browse_btn.grid(row=0, column=1, padx=(8, 0))

        self.hint_var = tk.StringVar()
        tb.Label(
            config_card, textvariable=self.hint_var, bootstyle=SECONDARY,
            font=("Segoe UI", 8)
        ).grid(row=6, column=0, sticky="w", pady=(6, 0))

        action_row = tb.Frame(self)
        action_row.pack(fill=X, pady=(0, 14))

        self.run_btn = tb.Button(
            action_row, text="\u25B6  Run", bootstyle=SUCCESS,
            command=self._start, width=14
        )
        self.run_btn.pack(side=LEFT)

        self.cancel_btn = tb.Button(
            action_row, text="\u25A0  Cancel", bootstyle=(DANGER, OUTLINE),
            command=self._cancel, state=DISABLED, width=12
        )
        self.cancel_btn.pack(side=LEFT, padx=(8, 0))

        self.move_btn = tb.Button(
            action_row, text="\U0001F4C1  Move Renamed Files...", bootstyle=(INFO, OUTLINE),
            command=self._move_successful, state=DISABLED
        )
        self.move_btn.pack(side=LEFT, padx=(8, 0))

        self.counts_var = tk.StringVar(value="")
        tb.Label(
            action_row, textvariable=self.counts_var,
            font=("Segoe UI", 9), bootstyle=SECONDARY
        ).pack(side=RIGHT)

        self.progress = tb.Progressbar(self, mode="determinate", bootstyle=(SUCCESS, STRIPED))
        self.progress.pack(fill=X, pady=(0, 14))

        log_card = tb.Labelframe(self, text="  Activity Log  ", padding=10)
        log_card.pack(fill=BOTH, expand=YES)

        log_container = tb.Frame(log_card)
        log_container.pack(fill=BOTH, expand=YES)

        self.log_box = tk.Text(
            log_container, wrap="word", state="disabled",
            font=("Consolas", 10), relief="flat", padx=10, pady=8, borderwidth=0
        )
        scroll = tb.Scrollbar(log_container, command=self.log_box.yview, bootstyle=ROUND)
        self.log_box.configure(yscrollcommand=scroll.set)
        self.log_box.pack(side=LEFT, fill=BOTH, expand=YES)
        scroll.pack(side=RIGHT, fill=Y)

    def apply_log_colors(self):
        palette = LOG_COLORS["dark"] if self.get_dark_mode() else LOG_COLORS["light"]
        self.log_box.configure(
            background=palette["bg"], foreground=palette["fg"], insertbackground=palette["fg"]
        )
        self.log_box.tag_configure("ok", foreground=palette["ok"])
        self.log_box.tag_configure("err", foreground=palette["err"])
        self.log_box.tag_configure("warn", foreground=palette["warn"])
        self.log_box.tag_configure("done", foreground=palette["done"], font=("Consolas", 10, "bold"))
        self.log_box.tag_configure("info", foreground=palette["info"])

    def _on_mode_change(self):
        if self.mode_var.get() == "copy":
            self.output_label.configure(state=NORMAL)
            self.output_entry.configure(state=NORMAL)
            self.output_browse_btn.configure(state=NORMAL)
            self.hint_var.set("Originals stay untouched. Renamed copies go to the output folder; unread barcodes go to output/failed/.")
        else:
            self.output_entry.configure(state=DISABLED)
            self.output_browse_btn.configure(state=DISABLED)
            self.hint_var.set("Files are renamed directly in the source folder. Unread barcodes are moved to source/failed/.")

    def _browse_source(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_var.set(folder)

    def _browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)

    def _append_log(self, text, tag="info"):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _start(self):
        source = self.source_var.get().strip()
        mode = self.mode_var.get()
        output = self.output_var.get().strip()

        if not source or not os.path.isdir(source):
            messagebox.showerror("Missing info", "Please choose a valid source folder.")
            return
        if mode == "copy" and not output:
            messagebox.showerror("Missing info", "Please choose an output folder for the copy mode.")
            return
        if mode == "copy" and os.path.abspath(output) == os.path.abspath(source):
            messagebox.showerror("Invalid folder", "Output folder must be different from the source folder.")
            return

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.progress["value"] = 0
        self.counts_var.set("")
        self.run_btn.configure(state=DISABLED)
        self.cancel_btn.configure(state=NORMAL)
        self.move_btn.configure(state=DISABLED)
        self.successful_paths = []

        self.stop_event.clear()
        self.worker = BarcodeRenameWorker(source, mode, output, self.msg_queue, self.stop_event)
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()

    def _move_successful(self):
        if not self.successful_paths:
            messagebox.showinfo("Nothing to move", "There are no successfully renamed files to move yet.")
            return

        target = filedialog.askdirectory(title="Choose destination folder for renamed files")
        if not target:
            return

        existing = [p for p in self.successful_paths if os.path.exists(p)]
        missing = len(self.successful_paths) - len(existing)

        if not existing:
            messagebox.showwarning("Nothing to move", "None of the previously renamed files could be found on disk anymore.")
            return

        moved, errors = 0, 0
        self._append_log(f"--- Moving {len(existing)} renamed file(s) to: {target} ---", "info")

        new_paths = []
        for path in existing:
            try:
                fname = os.path.basename(path)
                dest = unique_destination(target, os.path.splitext(fname)[0], os.path.splitext(fname)[1])
                shutil.move(path, dest)
                new_paths.append(dest)
                moved += 1
                self._append_log(f"\u2713 Moved: {fname}", "ok")
            except Exception as e:
                new_paths.append(path)
                errors += 1
                self._append_log(f"\u2717 Could not move {os.path.basename(path)}: {e}", "err")

        self.successful_paths = new_paths
        self._append_log(f"Move complete. Moved: {moved}   Errors: {errors}", "done")
        if missing:
            self._append_log(f"Note: {missing} previously tracked file(s) were no longer found.", "warn")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    text, tag = payload
                    self._append_log(text, tag)
                elif kind == "progress_max":
                    self.progress.configure(maximum=max(payload, 1))
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "counts":
                    renamed, failed, total = payload
                    self.counts_var.set(f"Renamed: {renamed}   Failed: {failed}   Total: {total}")
                elif kind == "success_path":
                    self.successful_paths.append(payload)
                elif kind == "finished":
                    self.run_btn.configure(state=NORMAL)
                    self.cancel_btn.configure(state=DISABLED)
                    self.move_btn.configure(state=NORMAL if self.successful_paths else DISABLED)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


# ========================================================= INVOICE OCR TAB

INVOICE_NUMBER_RE = re.compile(r"\d{5}")


def render_page_image(pdf_path, dpi, page_rotation_override=None):
    """Render page 1 of a PDF to a PIL image, correcting for the page's own
    /Rotate flag (PyMuPDF does not apply it to get_pixmap output by default)."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    rotation = page.rotation if page_rotation_override is None else page_rotation_override
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB" if pix.n >= 3 else "L", (pix.width, pix.height), pix.samples)
    doc.close()
    if rotation:
        img = img.rotate(-rotation, expand=True)
    return img


def crop_fractional(img, left_pct, top_pct, width_pct, height_pct):
    w, h = img.size
    l = int(w * left_pct / 100)
    t = int(h * top_pct / 100)
    ww = int(w * width_pct / 100)
    hh = int(h * height_pct / 100)
    l = max(0, min(l, w - 1))
    t = max(0, min(t, h - 1))
    r = max(l + 1, min(l + ww, w))
    b = max(t + 1, min(t + hh, h))
    return img.crop((l, t, r, b))


def match_prefixed_5digit(digits_only):
    """Matcher for the '7/ XXXXX' tax-invoice format: 5 digits right after a
    constant leading '7'."""
    m = re.search(r"7(\d{5})", digits_only)
    if m:
        return m.group(1)
    m = INVOICE_NUMBER_RE.search(digits_only)
    return m.group(0) if m else None


def match_9digit_run(digits_only):
    """Matcher for the internal service-invoice format: a plain 9-digit
    number (e.g. 140500035), no prefix character."""
    m = re.search(r"\d{9}", digits_only)
    if m:
        return m.group(0)
    # fall back to the longest digit run if we didn't get a clean 9
    runs = re.findall(r"\d+", digits_only)
    if runs:
        longest = max(runs, key=len)
        if len(longest) >= 6:
            return longest
    return None


def extract_invoice_number(pdf_path, settings, tesseract_cmd=None,
                            lang="fas", psm=6, matcher=match_prefixed_5digit):
    """Render, crop, optionally rotate the crop, OCR, and return
    (matched_number, raw_ocr_text, crop_image) or (None, raw_text, crop_image)."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    img = render_page_image(pdf_path, settings["dpi"], settings["page_rotation"])
    crop = crop_fractional(
        img, settings["left"], settings["top"], settings["width"], settings["height"]
    )
    if settings["crop_rotation"]:
        crop = crop.rotate(settings["crop_rotation"], expand=True)

    scale = settings.get("upscale", 1)
    if scale and scale != 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)

    raw_text = pytesseract.image_to_string(crop, lang=lang, config=f"--psm {psm}")
    normalized = normalize_digits(raw_text)

    # Collapse everything down to just the digit characters, ignoring spaces,
    # slashes, and any misread letters in between (OCR often inserts noise
    # between digits it otherwise reads correctly).
    digits_only = re.sub(r"\D", "", normalized)
    matched = matcher(digits_only)

    return matched, normalized, crop


class InvoiceOCRWorker(threading.Thread):
    def __init__(self, files, mode, output_folder, settings, tesseract_cmd, msg_queue, stop_event,
                 lang="fas", psm=6, matcher=match_prefixed_5digit, filename_builder=lambda d: "7" + d):
        super().__init__(daemon=True)
        self.files = files
        self.mode = mode
        self.output_folder = output_folder
        self.settings = settings
        self.tesseract_cmd = tesseract_cmd
        self.q = msg_queue
        self.stop_event = stop_event
        self.lang = lang
        self.psm = psm
        self.matcher = matcher
        self.filename_builder = filename_builder

    def log(self, text, tag="info"):
        self.q.put(("log", (text, tag)))

    def run(self):
        try:
            total = len(self.files)
            self.log(f"Processing {total} PDF file(s).")
            self.q.put(("progress_max", total))

            if self.mode == "copy":
                dest_root = self.output_folder
                failed_root = os.path.join(self.output_folder, "failed")
            else:
                dest_root = None  # per-file: same folder as the file itself
                failed_root = None

            os.makedirs(dest_root, exist_ok=True) if dest_root else None

            renamed, failed = 0, 0

            for idx, src_path in enumerate(self.files, start=1):
                if self.stop_event.is_set():
                    self.log("Cancelled by user.", "warn")
                    break

                fname = os.path.basename(src_path)
                matched = None

                try:
                    matched, raw_text, _ = extract_invoice_number(
                        src_path, self.settings, self.tesseract_cmd,
                        lang=self.lang, psm=self.psm, matcher=self.matcher
                    )
                except Exception as e:
                    self.log(f"\u2717 ERROR reading {fname}: {e}", "err")

                this_dest_root = dest_root if self.mode == "copy" else os.path.dirname(src_path)
                this_failed_root = (
                    failed_root if self.mode == "copy"
                    else os.path.join(os.path.dirname(src_path), "failed")
                )

                try:
                    if matched:
                        new_base = self.filename_builder(matched)
                        dest_path = unique_destination(this_dest_root, new_base, ".pdf")

                        if self.mode == "copy":
                            shutil.copy2(src_path, dest_path)
                        else:
                            os.rename(src_path, dest_path)

                        renamed += 1
                        self.log(f"\u2713 {fname}  \u2192  {os.path.basename(dest_path)}", "ok")
                        self.q.put(("success_path", dest_path))
                    else:
                        os.makedirs(this_failed_root, exist_ok=True)
                        dest_path = unique_destination(this_failed_root, os.path.splitext(fname)[0], ".pdf")

                        if self.mode == "copy":
                            shutil.copy2(src_path, dest_path)
                        else:
                            os.rename(src_path, dest_path)

                        failed += 1
                        self.log(f"\u2717 No number found: {fname}  \u2192  moved to failed/", "warn")

                except Exception as e:
                    failed += 1
                    self.log(f"\u2717 FAILED handling {fname}: {e}", "err")

                self.q.put(("progress", idx))
                self.q.put(("counts", (renamed, failed, total)))

            self.log(f"Done. Renamed: {renamed}  Failed: {failed}  Total: {total}", "done")

        except Exception as e:
            self.log(f"FATAL ERROR: {e}", "err")
        finally:
            self.q.put(("finished", None))


class InvoiceOCRTab(tb.Frame):
    """OCR-based renamer for a numeric field printed on the page (configurable
    crop defaults, OCR language, digit matcher, and filename format so this
    same class can drive multiple tabs for different document templates)."""

    def __init__(self, master, get_dark_mode, *,
                 lang="fas", psm=6, matcher=match_prefixed_5digit,
                 filename_builder=lambda d: "7" + d,
                 filename_preview_builder=None,
                 defaults=None,
                 files_hint="Select PDF files or a folder of PDFs to rename."):
        super().__init__(master, padding=16)
        self.get_dark_mode = get_dark_mode
        self.lang = lang
        self.psm = psm
        self.matcher = matcher
        self.filename_builder = filename_builder
        self.filename_preview_builder = filename_preview_builder or filename_builder
        self.defaults = defaults or {
            "left": 2.0, "top": 5.0, "width": 10.0, "height": 18.0,
            "page_rotation": 90, "crop_rotation": 90, "dpi": 400, "upscale": 2,
        }
        self.files_hint = files_hint

        self.msg_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.mode_var = tk.StringVar(value="copy")
        self.selected_files = []
        self.successful_paths = []
        self.preview_photo = None  # keep a reference so it isn't garbage collected

        self._build_ui()
        self._on_mode_change()
        self.apply_log_colors()

        if not HAVE_TESSERACT:
            self._append_log(
                "\u2717 pytesseract is not installed. Run: pip install pytesseract "
                "and install the Tesseract OCR engine separately.", "err"
            )

        self.after(100, self._poll_queue)

    def _build_ui(self):
        # ---- Source selection ----
        src_card = tb.Labelframe(self, text="  Files  ", padding=16)
        src_card.pack(fill=X, pady=(0, 12))

        btn_row = tb.Frame(src_card)
        btn_row.pack(fill=X)
        tb.Button(
            btn_row, text="Select PDF Files...", bootstyle=(SECONDARY, OUTLINE),
            command=self._select_files
        ).pack(side=LEFT)
        tb.Button(
            btn_row, text="Select Folder...", bootstyle=(SECONDARY, OUTLINE),
            command=self._select_folder
        ).pack(side=LEFT, padx=(8, 0))

        self.files_var = tk.StringVar(value=self.files_hint)
        tb.Label(src_card, textvariable=self.files_var, bootstyle=SECONDARY, font=("Segoe UI", 9)).pack(
            anchor="w", pady=(8, 0)
        )

        # ---- Mode ----
        mode_card = tb.Labelframe(self, text="  Output  ", padding=16)
        mode_card.pack(fill=X, pady=(0, 12))
        mode_card.columnconfigure(0, weight=1)

        mode_row = tb.Frame(mode_card)
        mode_row.grid(row=0, column=0, sticky="ew")
        tb.Radiobutton(
            mode_row, text="Rename in place (same folder as each file)", value="in_place",
            variable=self.mode_var, bootstyle="toolbutton", command=self._on_mode_change
        ).pack(side=LEFT, padx=(0, 8))
        tb.Radiobutton(
            mode_row, text="Copy renamed files to a new folder", value="copy",
            variable=self.mode_var, bootstyle="toolbutton", command=self._on_mode_change
        ).pack(side=LEFT)

        self.output_label = tb.Label(mode_card, text="Output folder", font=("Segoe UI", 9, "bold"))
        self.output_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
        out_row = tb.Frame(mode_card)
        out_row.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        out_row.columnconfigure(0, weight=1)

        self.output_var = tk.StringVar()
        self.output_entry = tb.Entry(out_row, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        self.output_browse_btn = tb.Button(
            out_row, text="Browse...", bootstyle=(SECONDARY, OUTLINE), command=self._browse_output
        )
        self.output_browse_btn.grid(row=0, column=1, padx=(8, 0))

        self.hint_var = tk.StringVar()
        tb.Label(mode_card, textvariable=self.hint_var, bootstyle=SECONDARY, font=("Segoe UI", 8)).grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )

        # ---- Calibration ----
        calib_card = tb.Labelframe(
            self, text="  Number Field Calibration  (use Preview to dial this in) ", padding=16
        )
        calib_card.pack(fill=X, pady=(0, 12))

        grid = tb.Frame(calib_card)
        grid.pack(fill=X)

        def spin(row, col, label, var, frm=0, to=100, width=6):
            tb.Label(grid, text=label, font=("Segoe UI", 8, "bold")).grid(
                row=row, column=col, sticky="w", padx=(0 if col == 0 else 14, 4)
            )
            sb = tb.Spinbox(grid, from_=frm, to=to, textvariable=var, width=width)
            sb.grid(row=row + 1, column=col, sticky="w", padx=(0 if col == 0 else 14, 4))
            return sb

        self.left_var = tk.DoubleVar(value=self.defaults["left"])
        self.top_var = tk.DoubleVar(value=self.defaults["top"])
        self.width_var = tk.DoubleVar(value=self.defaults["width"])
        self.height_var = tk.DoubleVar(value=self.defaults["height"])
        self.page_rot_var = tk.StringVar(value=str(self.defaults["page_rotation"]))
        self.crop_rot_var = tk.StringVar(value=str(self.defaults["crop_rotation"]))
        self.dpi_var = tk.IntVar(value=self.defaults["dpi"])

        spin(0, 0, "Left %", self.left_var)
        spin(0, 1, "Top %", self.top_var)
        spin(0, 2, "Width %", self.width_var)
        spin(0, 3, "Height %", self.height_var)

        tb.Label(grid, text="Page rotation", font=("Segoe UI", 8, "bold")).grid(
            row=0, column=4, sticky="w", padx=(14, 4)
        )
        tb.Combobox(
            grid, textvariable=self.page_rot_var, values=["0", "90", "180", "270"],
            width=5, state="readonly"
        ).grid(row=1, column=4, sticky="w", padx=(14, 4))

        tb.Label(grid, text="Extra rotation of crop", font=("Segoe UI", 8, "bold")).grid(
            row=0, column=5, sticky="w", padx=(14, 4)
        )
        tb.Combobox(
            grid, textvariable=self.crop_rot_var, values=["0", "90", "180", "270"],
            width=5, state="readonly"
        ).grid(row=1, column=5, sticky="w", padx=(14, 4))

        tessrow = tb.Frame(calib_card)
        tessrow.pack(fill=X, pady=(12, 0))
        tb.Label(tessrow, text="Tesseract path (leave blank if it's on PATH)", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tess_inner = tb.Frame(tessrow)
        tess_inner.pack(fill=X, pady=(2, 0))
        tess_inner.columnconfigure(0, weight=1)
        self.tesseract_path_var = tk.StringVar(
                value=BUNDLED_TESSERACT_PATH or ""
                        )
        tb.Entry(tess_inner, textvariable=self.tesseract_path_var).grid(row=0, column=0, sticky="ew")
        tb.Button(
            tess_inner, text="Browse...", bootstyle=(SECONDARY, OUTLINE),
            command=self._browse_tesseract
        ).grid(row=0, column=1, padx=(8, 0))

        preview_row = tb.Frame(calib_card)
        preview_row.pack(fill=X, pady=(12, 0))
        tb.Button(
            preview_row, text="\U0001F50D  Preview First File", bootstyle=(INFO, OUTLINE),
            command=self._preview
        ).pack(side=LEFT)

        self.preview_result_var = tk.StringVar(value="")
        tb.Label(
            preview_row, textvariable=self.preview_result_var, font=("Segoe UI", 9, "bold")
        ).pack(side=LEFT, padx=(12, 0))

        preview_panel = tb.Frame(calib_card)
        preview_panel.pack(fill=X, pady=(8, 0))

        self.preview_image_label = tb.Label(preview_panel, text="(no preview yet)", bootstyle=SECONDARY)
        self.preview_image_label.pack(side=LEFT, anchor="n")

        self.preview_text_box = tk.Text(preview_panel, height=6, width=40, font=("Consolas", 9))
        self.preview_text_box.pack(side=LEFT, padx=(12, 0), fill=X, expand=YES)
        self.preview_text_box.insert("1.0", "Raw OCR text will appear here after Preview.")
        self.preview_text_box.configure(state="disabled")

        # ---- Action row ----
        action_row = tb.Frame(self)
        action_row.pack(fill=X, pady=(0, 12))

        self.run_btn = tb.Button(
            action_row, text="\u25B6  Run", bootstyle=SUCCESS, command=self._start, width=14
        )
        self.run_btn.pack(side=LEFT)

        self.cancel_btn = tb.Button(
            action_row, text="\u25A0  Cancel", bootstyle=(DANGER, OUTLINE),
            command=self._cancel, state=DISABLED, width=12
        )
        self.cancel_btn.pack(side=LEFT, padx=(8, 0))

        self.move_btn = tb.Button(
            action_row, text="\U0001F4C1  Move Renamed Files...", bootstyle=(INFO, OUTLINE),
            command=self._move_successful, state=DISABLED
        )
        self.move_btn.pack(side=LEFT, padx=(8, 0))

        self.counts_var = tk.StringVar(value="")
        tb.Label(action_row, textvariable=self.counts_var, font=("Segoe UI", 9), bootstyle=SECONDARY).pack(
            side=RIGHT
        )

        self.progress = tb.Progressbar(self, mode="determinate", bootstyle=(SUCCESS, STRIPED))
        self.progress.pack(fill=X, pady=(0, 12))

        log_card = tb.Labelframe(self, text="  Activity Log  ", padding=10)
        log_card.pack(fill=BOTH, expand=YES)

        log_container = tb.Frame(log_card)
        log_container.pack(fill=BOTH, expand=YES)

        self.log_box = tk.Text(
            log_container, wrap="word", state="disabled",
            font=("Consolas", 10), relief="flat", padx=10, pady=8, borderwidth=0
        )
        scroll = tb.Scrollbar(log_container, command=self.log_box.yview, bootstyle=ROUND)
        self.log_box.configure(yscrollcommand=scroll.set)
        self.log_box.pack(side=LEFT, fill=BOTH, expand=YES)
        scroll.pack(side=RIGHT, fill=Y)

    def apply_log_colors(self):
        palette = LOG_COLORS["dark"] if self.get_dark_mode() else LOG_COLORS["light"]
        self.log_box.configure(
            background=palette["bg"], foreground=palette["fg"], insertbackground=palette["fg"]
        )
        self.log_box.tag_configure("ok", foreground=palette["ok"])
        self.log_box.tag_configure("err", foreground=palette["err"])
        self.log_box.tag_configure("warn", foreground=palette["warn"])
        self.log_box.tag_configure("done", foreground=palette["done"], font=("Consolas", 10, "bold"))
        self.log_box.tag_configure("info", foreground=palette["info"])
        self.preview_text_box.configure(background=palette["bg"], foreground=palette["fg"])

    def _on_mode_change(self):
        if self.mode_var.get() == "copy":
            self.output_label.configure(state=NORMAL)
            self.output_entry.configure(state=NORMAL)
            self.output_browse_btn.configure(state=NORMAL)
            self.hint_var.set("Originals stay untouched. Renamed copies go to the output folder; unread ones go to output/failed/.")
        else:
            self.output_entry.configure(state=DISABLED)
            self.output_browse_btn.configure(state=DISABLED)
            self.hint_var.set("Files are renamed directly where they are. Unread ones move to a failed/ subfolder next to each file.")

    def _select_files(self):
        paths = filedialog.askopenfilenames(title="Select PDF files", filetypes=[("PDF files", "*.pdf")])
        if paths:
            self.selected_files = list(paths)
            self.files_var.set(f"{len(self.selected_files)} file(s) selected.")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select a folder of PDFs")
        if folder:
            self.selected_files = list_pdfs_in_folder(folder)
            self.files_var.set(f"{len(self.selected_files)} PDF file(s) found in folder.")

    def _browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)

    def _browse_tesseract(self):
        path = filedialog.askopenfilename(title="Locate tesseract.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.tesseract_path_var.set(path)

    def _current_settings(self):
        return {
            "left": self.left_var.get(),
            "top": self.top_var.get(),
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "page_rotation": int(self.page_rot_var.get()),
            "crop_rotation": int(self.crop_rot_var.get()),
            "dpi": self.dpi_var.get(),
            "upscale": self.defaults.get("upscale", 2),
        }

    def _append_log(self, text, tag="info"):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _preview(self):
        if not HAVE_TESSERACT:
            messagebox.showerror("Missing dependency", "pytesseract is not installed. Run: pip install pytesseract")
            return
        if not self.selected_files:
            messagebox.showinfo("No files", "Select PDF files or a folder first.")
            return

        tess_cmd = self.tesseract_path_var.get().strip() or None
        sample = self.selected_files[0]

        try:
            matched, raw_text, crop = extract_invoice_number(
                sample, self._current_settings(), tess_cmd,
                lang=self.lang, psm=self.psm, matcher=self.matcher
            )
        except Exception as e:
            messagebox.showerror("Preview failed", str(e))
            return

        # show crop thumbnail
        thumb = crop.copy()
        thumb.thumbnail((260, 260))
        self.preview_photo = ImageTk.PhotoImage(thumb)
        self.preview_image_label.configure(image=self.preview_photo, text="")

        self.preview_text_box.configure(state="normal")
        self.preview_text_box.delete("1.0", "end")
        self.preview_text_box.insert("1.0", raw_text.strip() or "(no text recognized)")
        self.preview_text_box.configure(state="disabled")

        if matched:
            self.preview_result_var.set(f"Detected number: {self.filename_preview_builder(matched)}.pdf")
        else:
            self.preview_result_var.set("No number found \u2014 adjust the crop box above and preview again.")

    def _start(self):
        if not HAVE_TESSERACT:
            messagebox.showerror("Missing dependency", "pytesseract is not installed. Run: pip install pytesseract")
            return
        if not self.selected_files:
            messagebox.showerror("Missing info", "Select PDF files or a folder first.")
            return

        mode = self.mode_var.get()
        output = self.output_var.get().strip()
        if mode == "copy" and not output:
            messagebox.showerror("Missing info", "Please choose an output folder for the copy mode.")
            return

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.progress["value"] = 0
        self.counts_var.set("")
        self.run_btn.configure(state=DISABLED)
        self.cancel_btn.configure(state=NORMAL)
        self.move_btn.configure(state=DISABLED)
        self.successful_paths = []

        tess_cmd = self.tesseract_path_var.get().strip() or None

        self.stop_event.clear()
        self.worker = InvoiceOCRWorker(
            list(self.selected_files), mode, output, self._current_settings(),
            tess_cmd, self.msg_queue, self.stop_event,
            lang=self.lang, psm=self.psm, matcher=self.matcher,
            filename_builder=self.filename_builder
        )
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()

    def _move_successful(self):
        if not self.successful_paths:
            messagebox.showinfo("Nothing to move", "There are no successfully renamed files to move yet.")
            return

        target = filedialog.askdirectory(title="Choose destination folder for renamed files")
        if not target:
            return

        existing = [p for p in self.successful_paths if os.path.exists(p)]
        missing = len(self.successful_paths) - len(existing)

        if not existing:
            messagebox.showwarning("Nothing to move", "None of the previously renamed files could be found on disk anymore.")
            return

        moved, errors = 0, 0
        self._append_log(f"--- Moving {len(existing)} renamed file(s) to: {target} ---", "info")

        new_paths = []
        for path in existing:
            try:
                fname = os.path.basename(path)
                dest = unique_destination(target, os.path.splitext(fname)[0], os.path.splitext(fname)[1])
                shutil.move(path, dest)
                new_paths.append(dest)
                moved += 1
                self._append_log(f"\u2713 Moved: {fname}", "ok")
            except Exception as e:
                new_paths.append(path)
                errors += 1
                self._append_log(f"\u2717 Could not move {os.path.basename(path)}: {e}", "err")

        self.successful_paths = new_paths
        self._append_log(f"Move complete. Moved: {moved}   Errors: {errors}", "done")
        if missing:
            self._append_log(f"Note: {missing} previously tracked file(s) were no longer found.", "warn")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    text, tag = payload
                    self._append_log(text, tag)
                elif kind == "progress_max":
                    self.progress.configure(maximum=max(payload, 1))
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "counts":
                    renamed, failed, total = payload
                    self.counts_var.set(f"Renamed: {renamed}   Failed: {failed}   Total: {total}")
                elif kind == "success_path":
                    self.successful_paths.append(payload)
                elif kind == "finished":
                    self.run_btn.configure(state=NORMAL)
                    self.cancel_btn.configure(state=DISABLED)
                    self.move_btn.configure(state=NORMAL if self.successful_paths else DISABLED)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)



# ========================================================= INVOICE FINANCIAL TAB

class InvoiceFinancialTab(tb.Frame):
    """Batch-process scanned Iranian invoice PDFs: barcode -> invoice number,
    table rows, validate totals, rename, show results, export Excel."""

    def __init__(self, master, get_dark_mode):
        super().__init__(master, padding=16)
        self.get_dark_mode = get_dark_mode
        self.msg_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.entries = []
        self.model = None

        self._build_ui()
        self.apply_log_colors()
        self.after(100, self._poll_queue)

        if not HAVE_INVOICE_FINANCIAL:
            self._append_log(
                "invoice_financial module not available. Install opencv-python-headless openpyxl.", "err"
            )

    def _build_ui(self):
        src_card = tb.Labelframe(self, text="  Invoice Folder  ", padding=16)
        src_card.pack(fill=X, pady=(0, 12))

        btn_row = tb.Frame(src_card)
        btn_row.pack(fill=X)
        tb.Button(
            btn_row, text="Select Invoice Folder", bootstyle=(SECONDARY, OUTLINE),
            command=self._browse_folder
        ).pack(side=LEFT)

        self.folder_var = tk.StringVar(value="Choose a folder containing scanned invoice PDFs.")
        tb.Label(src_card, textvariable=self.folder_var, bootstyle=SECONDARY, font=("Segoe UI", 9)).pack(
            anchor="w", pady=(8, 0)
        )

        action_row = tb.Frame(self)
        action_row.pack(fill=X, pady=(0, 8))

        self.run_btn = tb.Button(
            action_row, text="\u25B6  Process Invoices", bootstyle=SUCCESS,
            command=self._start, width=18
        )
        self.run_btn.pack(side=LEFT)

        self.cancel_btn = tb.Button(
            action_row, text="\u25A0  Stop", bootstyle=(DANGER, OUTLINE),
            command=self._cancel, state=DISABLED, width=10
        )
        self.cancel_btn.pack(side=LEFT, padx=(8, 0))

        self.export_btn = tb.Button(
            action_row, text="Export Results to Excel", bootstyle=(INFO, OUTLINE),
            command=self._export, state=DISABLED
        )
        self.export_btn.pack(side=LEFT, padx=(8, 0))

        self.stats_var = tk.StringVar(value="Total PDFs: 0  Successfully processed: 0  Failed: 0")
        tb.Label(
            action_row, textvariable=self.stats_var,
            font=("Segoe UI", 9), bootstyle=SECONDARY
        ).pack(side=RIGHT)

        self.progress = tb.Progressbar(self, mode="determinate", bootstyle=(SUCCESS, STRIPED))
        self.progress.pack(fill=X, pady=(0, 10))

        sum_card = tb.Labelframe(self, text="  Summary  ", padding=10)
        sum_card.pack(fill=X, pady=(0, 10))
        self.summary_var = tk.StringVar(value="Total invoices: 0   Total rows: 0   Total debit: 0   Total credit: 0   Total balance: 0")
        tb.Label(sum_card, textvariable=self.summary_var, font=("Segoe UI", 9)).pack(anchor="w")

        res_card = tb.Labelframe(self, text="  Results (double-click Name to edit)  ", padding=8)
        res_card.pack(fill=BOTH, expand=YES, pady=(0, 8))

        cols = ("invoice", "row", "name", "ticket", "debit", "credit")
        self.tree = tb.Treeview(res_card, columns=cols, show="headings", height=8, bootstyle=PRIMARY)
        headings = {
            "invoice": ("Invoice", 80),
            "row": ("Row", 50),
            "name": ("Name", 220),
            "ticket": ("Ticket Number", 120),
            "debit": ("Debit", 110),
            "credit": ("Credit", 110),
        }
        for c, (h, w) in headings.items():
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center" if c != "name" else "w")

        yscroll = tb.Scrollbar(res_card, orient="vertical", command=self.tree.yview, bootstyle=ROUND)
        xscroll = tb.Scrollbar(res_card, orient="horizontal", command=self.tree.xview, bootstyle=ROUND)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        res_card.rowconfigure(0, weight=1)
        res_card.columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>", self._on_double_click)

        files_card = tb.Labelframe(self, text="  Files  ", padding=8)
        files_card.pack(fill=X, pady=(0, 8))

        fcols = ("orig", "new", "status", "reason")
        self.files_tree = tb.Treeview(files_card, columns=fcols, show="headings", height=4, bootstyle=SECONDARY)
        for c, h, w in (("orig", "Original file", 180), ("new", "New file", 100),
                        ("status", "Status", 80), ("reason", "Reason", 280)):
            self.files_tree.heading(c, text=h)
            self.files_tree.column(c, width=w, anchor="w")
        fyscroll = tb.Scrollbar(files_card, orient="vertical", command=self.files_tree.yview, bootstyle=ROUND)
        self.files_tree.configure(yscrollcommand=fyscroll.set)
        self.files_tree.pack(side=LEFT, fill=BOTH, expand=YES)
        fyscroll.pack(side=RIGHT, fill=Y)

        self.files_tree.tag_configure("SUCCESS", foreground="#198754")
        self.files_tree.tag_configure("WARNING", foreground="#fd7e14")
        self.files_tree.tag_configure("FAILED", foreground="#dc3545")

        log_card = tb.Labelframe(self, text="  Activity Log  ", padding=6)
        log_card.pack(fill=X)
        self.log_box = tk.Text(
            log_card, wrap="word", state="disabled", height=4,
            font=("Consolas", 9), relief="flat", padx=6, pady=4, borderwidth=0
        )
        self.log_box.pack(fill=X)

        note = ("Known limits: Name OCR is best-effort (not validated by totals). "
                "Amounts/tickets trained on sample invoices; mismatch -> FAILED, file left unrenamed. "
                "To support new layouts add labelled values in train_digits.py and rerun it.")
        tb.Label(self, text=note, bootstyle=SECONDARY, font=("Segoe UI", 8), wraplength=900).pack(
            anchor="w", pady=(6, 0)
        )

    def apply_log_colors(self):
        palette = LOG_COLORS["dark"] if self.get_dark_mode() else LOG_COLORS["light"]
        self.log_box.configure(
            bg=palette["bg"], fg=palette["fg"],
            insertbackground=palette["fg"],
        )
        for tag, key in (("ok", "ok"), ("err", "err"), ("warn", "warn"),
                         ("done", "done"), ("info", "info")):
            self.log_box.tag_configure(tag, foreground=palette[key])
        if self.get_dark_mode():
            self.files_tree.tag_configure("SUCCESS", foreground="#57e389")
            self.files_tree.tag_configure("WARNING", foreground="#ffb454")
            self.files_tree.tag_configure("FAILED", foreground="#ff6b6b")
        else:
            self.files_tree.tag_configure("SUCCESS", foreground="#198754")
            self.files_tree.tag_configure("WARNING", foreground="#fd7e14")
            self.files_tree.tag_configure("FAILED", foreground="#dc3545")

    def _append_log(self, text, tag="info"):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select folder of invoice PDFs")
        if path:
            self.folder_var.set(path)

    def _start(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder) or folder.startswith("Choose"):
            messagebox.showerror("Missing info", "Please select a valid invoice folder.")
            return
        if not HAVE_INVOICE_FINANCIAL:
            messagebox.showerror("Missing module", "invoice_financial is not available.")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        self.entries = []
        self.summary_var.set("Total invoices: 0   Total rows: 0   Total debit: 0   Total credit: 0   Total balance: 0")
        self.stats_var.set("Total PDFs: 0  Successfully processed: 0  Failed: 0")
        self.progress["value"] = 0
        self.export_btn.configure(state=DISABLED)

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.run_btn.configure(state=DISABLED)
        self.cancel_btn.configure(state=NORMAL)
        self.stop_event.clear()

        def progress_cb(i, total, entry):
            self.msg_queue.put(("progress", (i, total, entry)))

        def worker():
            try:
                if self.model is None:
                    self.model = DigitModel()
                if HAVE_TESSERACT and BUNDLED_TESSERACT_PATH:
                    pytesseract.pytesseract.tesseract_cmd = BUNDLED_TESSERACT_PATH
                entries = process_folder(
                    folder, model=self.model,
                    progress=progress_cb,
                    should_stop=lambda: self.stop_event.is_set()
                )
                self.msg_queue.put(("finished", entries))
            except Exception as e:
                self.msg_queue.put(("error", str(e)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _cancel(self):
        self.stop_event.set()
        self._append_log("Stop requested...", "warn")

    def _export(self):
        if not self.entries:
            messagebox.showinfo("Nothing to export", "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Excel results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            export_excel(path, self.entries)
            self._append_log(f"Exported to {path}", "done")
            messagebox.showinfo("Exported", f"Results saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            self._append_log(f"Export failed: {e}", "err")

    def _on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not item or col != "#3":
            return
        vals = list(self.tree.item(item, "values"))
        old_name = vals[2]
        dlg = tb.Toplevel(self)
        dlg.title("Edit Name")
        dlg.geometry("400x120")
        dlg.transient(self.winfo_toplevel())
        tb.Label(dlg, text="Correct the passenger name:").pack(pady=(12, 4), padx=12, anchor="w")
        var = tk.StringVar(value=old_name)
        ent = tb.Entry(dlg, textvariable=var, width=50)
        ent.pack(padx=12, fill=X)
        ent.focus_set()
        ent.select_range(0, "end")

        def save():
            new_name = var.get().strip()
            vals[2] = new_name
            self.tree.item(item, values=vals)
            inv = str(vals[0])
            rowno = int(vals[1])
            for e in self.entries:
                for r in e.get("rows", []):
                    if str(r["invoice"]) == inv and r["row"] == rowno:
                        r["name"] = new_name
                        break
            dlg.destroy()

        btnf = tb.Frame(dlg)
        btnf.pack(pady=10)
        tb.Button(btnf, text="Save", bootstyle=SUCCESS, command=save).pack(side=LEFT, padx=4)
        tb.Button(btnf, text="Cancel", bootstyle=SECONDARY, command=dlg.destroy).pack(side=LEFT, padx=4)
        dlg.bind("<Return>", lambda e: save())

    def _fmt_num(self, n):
        try:
            return f"{int(n):,}"
        except Exception:
            return str(n)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "progress":
                    i, total, entry = payload
                    self.progress.configure(maximum=max(total, 1))
                    self.progress["value"] = i
                    tag = entry.get("status", "FAILED")
                    self.files_tree.insert(
                        "", "end",
                        values=(entry.get("file", ""), entry.get("new_name", ""),
                                entry.get("status", ""), entry.get("reason", "")),
                        tags=(tag,)
                    )
                    for r in entry.get("rows", []):
                        self.tree.insert(
                            "", "end",
                            values=(r["invoice"], r["row"], r["name"], r["ticket"],
                                    self._fmt_num(r["debit"]), self._fmt_num(r["credit"]))
                        )
                    self.entries.append(entry)
                    sm = summarize(self.entries)
                    self.stats_var.set(
                        f"Total PDFs: {sm['total']}  Successfully processed: {sm['ok']}  Failed: {sm['failed']}"
                    )
                    self.summary_var.set(
                        f"Total invoices: {sm['invoices']}   Total rows: {sm['rows']}   "
                        f"Total debit: {self._fmt_num(sm['debit'])}   "
                        f"Total credit: {self._fmt_num(sm['credit'])}   "
                        f"Total balance: {self._fmt_num(sm['balance'])}"
                    )
                    st = entry.get("status", "")
                    tag = "ok" if st == "SUCCESS" else ("warn" if st == "WARNING" else "err")
                    self._append_log(
                        f"{entry.get('file')}: {st} {entry.get('reason','')} -> {entry.get('new_name') or '-'}",
                        tag
                    )
                elif kind == "finished":
                    self.entries = payload
                    self.run_btn.configure(state=NORMAL)
                    self.cancel_btn.configure(state=DISABLED)
                    self.export_btn.configure(state=NORMAL if self.entries else DISABLED)
                    self._append_log("Processing finished.", "done")
                elif kind == "error":
                    self.run_btn.configure(state=NORMAL)
                    self.cancel_btn.configure(state=DISABLED)
                    self._append_log(f"Error: {payload}", "err")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


# ======================================================================= APP

class App(tb.Window):
    def __init__(self):
        super().__init__(themename=LIGHT_THEME)
        self.title("PDF Renamer")
        self.geometry("980x760")
        self.minsize(820, 620)

        self.dark_mode = tk.BooleanVar(value=False)

        outer = tb.Frame(self, padding=20)
        outer.pack(fill=BOTH, expand=YES)

        header = tb.Frame(outer)
        header.pack(fill=X, pady=(0, 12))

        tb.Label(header, text="\U0001F4C4 PDF Renamer", font=("Segoe UI", 18, "bold")).pack(side=LEFT)

        self.theme_toggle = tb.Checkbutton(
            header, text="\U0001F319  Dark mode", variable=self.dark_mode,
            bootstyle="round-toggle", command=self._toggle_theme
        )
        self.theme_toggle.pack(side=RIGHT)

        notebook = tb.Notebook(outer)
        notebook.pack(fill=BOTH, expand=YES)

        self.barcode_tab = BarcodeTab(notebook, self._get_dark_mode)
        self.invoice_tab = InvoiceOCRTab(notebook, self._get_dark_mode)
        self.internal_invoice_tab = InvoiceOCRTab(
            notebook, self._get_dark_mode,
            lang="eng", psm=6, matcher=match_9digit_run,
            filename_builder=lambda d: d,
            defaults={
                "left": 10.0, "top": 4.0, "width": 16.0, "height": 6.0,
                "page_rotation": 0, "crop_rotation": 0, "dpi": 400, "upscale": 1,
            },
            files_hint="Select the internal service-invoice PDFs (or a folder of them) to rename."
        )
        self.financial_tab = InvoiceFinancialTab(notebook, self._get_dark_mode)

        notebook.add(self.barcode_tab, text="  Barcode Rename  ")
        notebook.add(self.invoice_tab, text="  Invoice Number (OCR)  ")
        notebook.add(self.internal_invoice_tab, text="  Internal Invoice (OCR)  ")
        notebook.add(self.financial_tab, text="  Invoice Financial Data  ")

    def _get_dark_mode(self):
        return self.dark_mode.get()

    def _toggle_theme(self):
        self.style.theme_use(DARK_THEME if self.dark_mode.get() else LIGHT_THEME)
        self.barcode_tab.apply_log_colors()
        self.invoice_tab.apply_log_colors()
        self.internal_invoice_tab.apply_log_colors()
        self.financial_tab.apply_log_colors()


if __name__ == "__main__":
    app = App()
    app.mainloop()
