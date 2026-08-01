"""Download the NeonTreeEvaluation benchmark (annotated RGB tiles + VOC-XML).

Cross-dataset test for Direction A: US NEON forests (airborne RGB, 22 sites,
different biome/sensor than the Mediterranean VHRTrees) with box annotations.
Only the tiles that HAVE annotations are fetched (not all 1000 RGB tiles).

Run: .venv/bin/python -m improvement.neon.download
Writes data_neon/{rgb/*.tif, ann/*.xml, index.json}.
"""

import json
import os
import subprocess
import urllib.request

RAW = "https://raw.githubusercontent.com/weecology/NeonTreeEvaluation/master"
OUT = "data_neon"


def _gh_list(path):
    """List a repo dir via gh api (avoids GitHub API rate limits w/ auth)."""
    r = subprocess.run(["gh", "api", f"repos/weecology/NeonTreeEvaluation/contents/{path}"],
                       capture_output=True, text=True)
    return json.loads(r.stdout)


def _fetch(url, dest):
    if os.path.exists(dest):
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAIL {os.path.basename(dest)}: {e}")
        return False


def main():
    os.makedirs(f"{OUT}/rgb", exist_ok=True)
    os.makedirs(f"{OUT}/ann", exist_ok=True)

    anns = _gh_list("annotations")
    xml_names = [a["name"] for a in anns if a["name"].endswith(".xml")]
    print(f"annotation files: {len(xml_names)}")

    # RGB tiles present in the repo (filename stem must match an annotation)
    rgb = _gh_list("evaluation/RGB")
    rgb_by_stem = {os.path.splitext(f["name"])[0]: f["name"]
                   for f in rgb if f["name"].endswith(".tif")}

    index = []
    got = 0
    for xml in xml_names:
        stem = os.path.splitext(xml)[0]
        if stem not in rgb_by_stem:
            continue  # annotation without a matching RGB tile (skip)
        tif = rgb_by_stem[stem]
        ok_x = _fetch(f"{RAW}/annotations/{xml}", f"{OUT}/ann/{xml}")
        ok_i = _fetch(f"{RAW}/evaluation/RGB/{tif}", f"{OUT}/rgb/{tif}")
        if ok_x and ok_i:
            index.append({"stem": stem, "xml": xml, "tif": tif})
            got += 1
            if got % 20 == 0:
                print(f"  fetched {got} tiles...")
    json.dump(index, open(f"{OUT}/index.json", "w"), indent=2)
    print(f"done: {got} matched (rgb+annotation) tiles -> {OUT}/")


if __name__ == "__main__":
    main()
