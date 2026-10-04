# TASK (autonomous, no questions)

You are working in the folder where BarcodeRenamer.py lives (Python, ttkbootstrap, PyMuPDF, zxing-cpp, pytesseract, Pillow, numpy; PyInstaller specs BarcodeRenamer.spec / PDFRenamer.spec). Deliver a finished, working application. Do NOT ask me any question. If something is unclear, choose the simplest reasonable option, continue, and list your assumptions in the final report.

## Goal
Add ONE new tab, "Invoice Financial Data", to the existing app. It batch-processes scanned Iranian invoice PDFs: the user picks a folder; each PDF is read (barcode -> invoice number, table -> rows), validated against the printed totals, renamed to <invoice>.pdf, and shown in a results table that can be exported to Excel.

## Already done (do NOT rewrite, only fix real bugs)
These files are already in the folder, written and tested:
- invoice_financial.py : all logic, no tkinter. Use these functions: DigitModel, process_pdf, process_folder(folder, model, progress, should_stop), summarize(entries), export_excel(path, entries), rename_invoice.
- digit_model.npz : trained digit classifier (must ship with the exe).
- train_digits.py : rebuilds digit_model.npz from labelled samples.
- test_invoice.py : acceptance test, needs samples\inv86.pdf and samples\inv87.pdf. Must print OVERALL: PASS.
process_folder returns a list of dicts: file, status (SUCCESS/WARNING/FAILED), reason, invoice, new_name, rows (dicts: invoice,row,name,ticket,debit,credit). It already renames only on SUCCESS/WARNING, never overwrites, and reports duplicates.

## What you must build
1. In BarcodeRenamer.py add class InvoiceFinancialTab, written in the same style as the existing tabs (same frame/card layout, dark-mode handling, apply_log_colors, worker thread + queue + _poll_queue). Register it in App next to the other tabs. Do not change any existing tab or behavior.
2. Controls: button "Select Invoice Folder" + path label; "Process Invoices"; "Stop"; progress bar; stats line "Total PDFs: N  Successfully processed: N  Failed: N"; button "Export Results to Excel" (asks for save path, calls export_excel).
3. Results table (Treeview, large, scrollable): Invoice, Row, Name, Ticket Number, Debit, Credit, with thousands separators. Double-click a Name cell lets the user correct it; export must use the corrected value.
4. Files table: Original file, New file, Status, Reason (colour SUCCESS green, WARNING amber, FAILED red).
5. Summary block: Total invoices, Total rows, Total debit, Total credit, Total balance.
6. Before calling the library make sure pytesseract uses the app's existing Tesseract path / tessdata (the English language data is needed for names), exactly as the existing OCR tab sets it up.
7. Add to requirements: opencv-python-headless, openpyxl (and any missing). Update BOTH PyInstaller spec files: include digit_model.npz in datas, add cv2/openpyxl hidden imports if needed.
8. Everything offline, no cloud APIs, nothing uploaded.

## Verification (do all, show real output)
- python -c "import BarcodeRenamer" succeeds.
- python test_invoice.py prints OVERALL: PASS (invoice 86: 18 rows, debit 1,937,375,000, credit 0, balance 1,937,375,000, renamed 71818.pdf; invoice 87: 6 rows, debit 553,129,100, credit 132,000,000, balance 421,129,100, renamed 72774.pdf). If samples are missing, say so and skip only that test.
- Run the new tab's processing path on a temp copy of the two samples (not the originals) and confirm the table, summary, and Excel file (3 sheets: Details, Summary, Files).
- Build the exe with PyInstaller, start it once, confirm it opens with the new tab. If the GUI cannot be run in your environment, say exactly what you could not verify.

## Rules
- Never claim something works unless you ran it and saw the output.
- Do not retrain or change the digit model unless test_invoice.py fails; if you retrain, only through train_digits.py.
- Known limits to keep in the UI help/readme, not hide: Name OCR is best-effort (not covered by the totals check); amounts and tickets are validated/trained on two sample invoices, so a misread amount shows as FAILED "Financial total mismatch" and the file is left unrenamed; to support new invoices add their labelled values to train_digits.py and rerun it.

## Final report (short)
List every file modified or added, what changed, the real test outputs, and any assumption or unverified item.
