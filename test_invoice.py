"""Acceptance test on the two sample invoices. Run: python test_invoice.py"""
import os, shutil, tempfile, sys
from invoice_financial import process_pdf, rename_invoice, DigitModel

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = {
    "inv86.pdf": dict(invoice="71818", n=18, debit=1937375000, credit=0, balance=1937375000,
                      amounts=[123000000, 110000000, 103550000, 72059000, 84000000, 104000000, 75470000,
                               116330000, 130115000, 86260000, 75000000, 115500000, 75071000, 111520000,
                               159500000, 132000000, 132000000, 132000000]),
    "inv87.pdf": dict(invoice="72774", n=6, debit=553129100, credit=132000000, balance=421129100,
                      amounts=[104299100, 132000000, 116330000, 107000000, 102000000, 123500000],
                      credits=[0, 132000000, 0, 0, 0, 0],
                      tickets=["1013-639-677", "0009-255-573", "1410-596-582", "0001-056-092",
                               "0005-115-422", "0008-876-199"],
                      names=["MEHDI JAVADIMAJAREH", "BAHMAN LALEHZARZADEH", "SAADATMANDI/SEYYEDALI MR",
                             "NIMA FIROUZANBALASI", "NIMA FIROUZANBALASI", "SEYYEDALI SAADATMANDI"]),
}


def main():
    tmp = tempfile.mkdtemp()
    model = DigitModel()
    allok = True
    for fname, exp in EXPECT.items():
        src = os.path.join(tmp, "Adobe Scan " + fname)
        shutil.copy(os.path.join(HERE, "samples", fname), src)
        r = process_pdf(src, model)
        print(f"\n=== {fname}: status={r.status} reason={r.reason!r} invoice={r.invoice}")
        for row in r.rows:
            print(f"  {row['row']:>2} {row['name']:<28} {row['ticket']:<13} debit={row['debit']:>12,} credit={row['credit']:>12,}")
        d = sum(x["debit"] for x in r.rows); c = sum(x["credit"] for x in r.rows)
        print(f"  rows={len(r.rows)} debit={d:,} credit={c:,} balance={d - c:,}   printed={r.totals}")
        checks = {
            "invoice number": r.invoice == exp["invoice"],
            "row count": len(r.rows) == exp["n"],
            "debit total": d == exp["debit"],
            "credit total": c == exp["credit"],
            "balance": d - c == exp["balance"],
            "row amounts": [x["debit"] or x["credit"] for x in r.rows] == exp["amounts"] or
                           [x["debit"] for x in r.rows] == exp["amounts"],
            "status": r.status in ("SUCCESS", "WARNING"),
        }
        if "tickets" in exp:
            checks["tickets"] = [x["ticket"] for x in r.rows] == exp["tickets"]
            import re as _re
            nm = lambda v: _re.sub(r"\s+", " ", _re.sub(r"[/.\-]", " ", v)).strip()
            n_ok = sum(nm(a["name"]) == nm(b) for a, b in zip(r.rows, exp["names"]))
            print(f"  [INFO] names exact (best-effort OCR, not validated by totals): {n_ok}/{len(exp['names'])}")
            checks["credit rows"] = [x["credit"] for x in r.rows] == exp["credits"]
        if r.status in ("SUCCESS", "WARNING"):
            new, msg = rename_invoice(src, r.invoice)
            checks["renamed to " + r.invoice + ".pdf"] = bool(new) and os.path.basename(new) == r.invoice + ".pdf" and os.path.exists(new)
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
            allok &= v
    print("\nOVERALL:", "PASS" if allok else "FAIL")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
