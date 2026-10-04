"""Builds digit_model.npz from labelled sample invoices (run once, or again when you add
more labelled samples).  Usage:  python train_digits.py
Each entry below gives the known-correct values of an invoice; the glyph images are cut
from samples/<file>.pdf and stored as the classifier's training set."""
import os
import numpy as np
import fitz
from invoice_financial import (render_page, find_grid, extract_rows, find_total_boxes, _components,
                               glyph_feature, C_DEBIT, C_CREDIT, C_TICKET)

LABELLED = {
    "samples/inv86.pdf": dict(
        debit=[123000000, 110000000, 103550000, 72059000, 84000000, 104000000, 75470000, 116330000,
               130115000, 86260000, 75000000, 115500000, 75071000, 111520000, 159500000, 132000000,
               132000000, 132000000],
        credit=[0] * 18,
        tickets=["0001-389-155", "0003-243-981", "2400-532-193", "9573-333-964", "0001-391-984",
                 "0008-827-741", "2448-242-685", "2449-151-535", "6110-440-662", "9573-334-290",
                 "0001-057-094", "0005-082-328", "2448-242-689", "2448-242-690", "0003-802-092",
                 "0008-845-254", "0009-255-573", "0001-628-118"],
        boxes=dict(debit=1937375000, credit=0, balance=1937375000)),
    "samples/inv87.pdf": dict(
        debit=[104299100, 0, 116330000, 107000000, 102000000, 123500000],
        credit=[0, 132000000, 0, 0, 0, 0],
        tickets=["1013-639-677", "0009-255-573", "1410-596-582", "0001-056-092", "0005-115-422",
                 "0008-876-199"],
        boxes=dict(debit=553129100, credit=132000000, balance=421129100)),
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"cell": ([], []), "big": ([], [])}
    for rel, lab in LABELLED.items():
        doc = fitz.open(os.path.join(here, rel))
        g = render_page(doc, 0)
        rows, _ = extract_rows(g, find_grid(g))
        assert len(rows) == len(lab["debit"]), (rel, len(rows))
        for i, r in enumerate(rows):
            for ci, key in ((C_DEBIT, "debit"), (C_CREDIT, "credit")):
                v = lab[key][i]
                if v == 0 and False:
                    continue
                bw, comps = _components(r["cells"][ci], 10)
                digs = [b for b in comps if not (b[2] <= 8 and b[3] <= 14)]
                s = str(v)
                assert len(digs) == len(s), (rel, i, key, len(digs), s)
                for b, ch in zip(digs, s):
                    out["cell"][0].append(glyph_feature(bw, b)); out["cell"][1].append(ch)
            bw, comps = _components(r["cells"][C_TICKET], 10)
            digs = [b for b in comps if not (b[2] >= 2.5 * b[3])]
            t = lab["tickets"][i].replace("-", "")
            if len(digs) == 10:                       # skip rows with merged glyphs
                for b, ch in zip(digs, t):
                    out["big"][0].append(glyph_feature(bw, b)); out["big"][1].append(ch)
        gl = render_page(doc, len(doc) - 1)
        boxes = find_total_boxes(gl)
        for nm, v in lab["boxes"].items():
            if v == 0:
                continue
            bw, comps = _components(boxes[nm], 14)
            s = str(v)
            digs = [b for k, b in enumerate(comps[::-1]) if k % 4 != 3][::-1]
            assert len(digs) == len(s), (rel, nm)
            for b, ch in zip(digs, s):
                out["big"][0].append(glyph_feature(bw, b)); out["big"][1].append(ch)
    np.savez(os.path.join(here, "digit_model.npz"),
             X_cell=np.array(out["cell"][0]), Y_cell=np.array(out["cell"][1]),
             X_big=np.array(out["big"][0]), Y_big=np.array(out["big"][1]))
    print("cell glyphs:", len(out["cell"][1]), " big glyphs:", len(out["big"][1]))


if __name__ == "__main__":
    main()
