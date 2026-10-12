import json, glob
from squarewave.evaluate import evaluate
from squarewave.remote import RemoteOracle
ro = RemoteOracle()
files = sorted(glob.glob("results/uniform_beam/*.json"))
d = [json.load(open(f)) for f in files]
rows = evaluate(ro, [x["seq"] for x in d])
for f, x, r in zip(files, d, rows):
    print(f.split("/")[-1], "tok/bp", x["tok_per_bp"], "loss %.2f auroc %.3f (E %.3f B %.3f) icc %.3f heldout %.3f gc_wave_corr %.2f" % (r["loss"], r["auroc"], r["auroc_enformer"], r["auroc_borzoi"], r["icc"], r["heldout_auroc"], r["gc_wave_corr"]))
import pathlib
pathlib.Path("results/ubeam_eval.json").write_text(json.dumps(rows, indent=1))
