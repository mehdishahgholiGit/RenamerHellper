# RenamerHellper (PDF Renamer)

Desktop app for batch-renaming scanned PDFs.

```
python BarcodeRenamer.py
```

Install Python packages with:

```
pip install -r requirements.txt
```

Tesseract OCR is also required (not a pip package). The app looks for:

1. A bundled `tesseract\tesseract.exe` next to the script (or inside the packaged exe)
2. `C:\Program Files\Tesseract-OCR\tesseract.exe`

Windows installer: https://github.com/UB-Mannheim/tesseract/wiki  
During install, add the Persian (`fas`) language if you use the Invoice Number (OCR) tab.

## Tabs

- **Barcode Rename** — decode a barcode on the page and rename the PDF to that value
- **Invoice Number (OCR)** — crop/OCR a `7/ XXXXX` tax-invoice number
- **Internal Invoice (OCR)** — crop/OCR a 9-digit internal invoice number
- **Invoice Financial Data** — read barcode + financial table, validate printed totals, rename to `<invoice>.pdf`, export Excel

## Invoice Financial Data notes

- Name OCR is best-effort and is not covered by the totals check
- Amounts and tickets are validated/trained on the sample invoices in `samples\`
- A misread amount is reported as FAILED `Financial total mismatch` and the file is left unrenamed
- To support a new layout, add labelled values in `train_digits.py` and rerun it

## Tests

```
python test_invoice.py
```

Expected: `OVERALL: PASS` using `samples\inv86.pdf` and `samples\inv87.pdf`.
