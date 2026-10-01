"""Write py3langid's model as the flat file the Rust crawler reads, so both crawlers judge language alike."""

import struct
import sys

import numpy as np
import py3langid.langid as langid

model = langid.LanguageIdentifier.from_model_file(langid.MODEL_FILE)
ptc = np.asarray(model.nb_ptc)
pc = np.asarray(model.nb_pc)
classes = [str(c) for c in model.nb_classes]
nextmove = np.asarray(model.tk_nextmove)
rows = np.asarray(model.tk_row)
out = np.asarray(model.tk_output)
if ptc.dtype != np.float16 or ptc.shape != (len(out) and ptc.shape[0], len(classes)):
    sys.exit(f"unexpected model layout: ptc {ptc.dtype} {ptc.shape}, {len(classes)} classes")
with open(sys.argv[1], "wb") as target:
    target.write(b"LANGID01")
    target.write(struct.pack("<4I", ptc.shape[0], len(classes), len(nextmove), len(rows)))
    for code in classes:
        target.write(code.encode("ascii").ljust(4, b"\0"))
    target.write(pc.astype("<f4").tobytes())
    target.write(np.ascontiguousarray(ptc).astype("<f2").tobytes())
    target.write(nextmove.astype("<u4").tobytes())
    target.write(rows.astype("<u2").tobytes())
    target.write(out.astype("<i4").tobytes())
print(f"{ptc.shape[0]} features, {len(classes)} classes, {len(rows)} states written to {sys.argv[1]}")
