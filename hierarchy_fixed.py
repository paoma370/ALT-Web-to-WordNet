import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
mapping_path = r"C:\Users\Paola Marongiu\Documents\ALT\LREC\agricoltura-WN - mapping_completo.csv"
lmf_path = r"C:\Users\Paola Marongiu\Documents\ALT\LREC\wnita.lmf.gz"

# Output
direct_csv_path = r"C:\Users\Paola Marongiu\Documents\ALT\LREC\parole_per_depth_direct_tpdomO.csv"
extended_csv_path = r"C:\Users\Paola Marongiu\Documents\ALT\LREC\parole_per_depth_extended_tpdomO.csv"

# ---------------------------
# 1) Load + filter mapping
# ---------------------------
df = pd.read_csv(mapping_path, encoding="latin1", sep=None, engine="python")
df_O = df[df["semanticnet_tpdom"] == "O"].copy()

# ---------------------------
# 2) Parse WordNet LMF and compute depths using relType="has_hyperonym"
# ---------------------------
hypernym_rel = defaultdict(list)
all_synsets = set()

with gzip.open(lmf_path, "rb") as f:
    tree = ET.parse(f)
    root = tree.getroot()

for synset in root.iterfind(".//{*}Synset"):
    synset_id = synset.attrib.get("id")
    if synset_id:
        all_synsets.add(synset_id)
        for rel in synset.findall(".//{*}SynsetRelation"):
            if rel.attrib.get("relType") == "has_hyperonym":
                target = rel.attrib.get("target")
                if target:
                    hypernym_rel[synset_id].append(target)

# Reverse graph (parent -> children)
reverse_rel = defaultdict(list)
for child, parents in hypernym_rel.items():
    for p in parents:
        reverse_rel[p].append(child)

# Roots: synsets with no hyperonyms
roots = {s for s in all_synsets if s not in hypernym_rel}

# BFS depth from roots
depth = {}
queue = deque(roots)
for r in roots:
    depth[r] = 0

while queue:
    current = queue.popleft()
    for child in reverse_rel[current]:
        if child not in depth:
            depth[child] = depth[current] + 1
            queue.append(child)

# ---------------------------
# 3) Build parola -> synsets (direct / closest)
# ---------------------------
parola_to_direct = (
    df_O.dropna(subset=["semanticnet_parola_WN"])
        .groupby("semanticnet_parola")["semanticnet_parola_WN"]
        .apply(lambda s: set(s.dropna().unique()))
        .to_dict()
)

parola_to_closest = (
    df_O.dropna(subset=["semanticnet_parola_WN_closest"])
        .groupby("semanticnet_parola")["semanticnet_parola_WN_closest"]
        .apply(lambda s: set(s.dropna().unique()))
        .to_dict()
)

all_parole = sorted(df_O["semanticnet_parola"].dropna().unique().tolist())

# ---------------------------
# 4) Helper: long-form table builder
#    - mapping_source: "direct" or "closest"
#    - depth_adjusted: depth_wn (+1 ONLY if closest)
# ---------------------------
def rows_for_mapping(parola: str, synsets: set, mapping_source: str):
    out = []
    for syn in sorted(synsets):
        if syn in depth:  # only if synset is in WordNet graph
            depth_wn = int(depth[syn])
            depth_adjusted = depth_wn + 1 if mapping_source == "closest" else depth_wn
            out.append({
                "semanticnet_parola": parola,
                "synset_id": syn,
                "mapping_source": mapping_source,
                "depth_wn": depth_wn,
                "depth_adjusted": depth_adjusted
            })
    return out

# ---------------------------
# 5) DIRECT table (only direct links)
# ---------------------------
direct_rows = []
for parola, syns in parola_to_direct.items():
    direct_rows.extend(rows_for_mapping(parola, syns, "direct"))

direct_table = pd.DataFrame(direct_rows)
if len(direct_table) > 0:
    direct_table = direct_table.sort_values(
        ["depth_adjusted", "semanticnet_parola", "synset_id"]
    ).reset_index(drop=True)

direct_table.to_csv(direct_csv_path, index=False, encoding="utf-8")
print("Saved direct CSV:", direct_csv_path)

# ---------------------------
# 6) EXTENDED table:
#    if parola has direct -> use direct
#    else if has closest -> use closest (depth +1)
# ---------------------------
extended_rows = []
for parola in all_parole:
    if parola in parola_to_direct and len(parola_to_direct[parola]) > 0:
        extended_rows.extend(rows_for_mapping(parola, parola_to_direct[parola], "direct"))
    else:
        syns_cl = parola_to_closest.get(parola, set())
        if len(syns_cl) > 0:
            extended_rows.extend(rows_for_mapping(parola, syns_cl, "closest"))

extended_table = pd.DataFrame(extended_rows)
if len(extended_table) > 0:
    extended_table = extended_table.sort_values(
        ["depth_adjusted", "semanticnet_parola", "synset_id"]
    ).reset_index(drop=True)

extended_table.to_csv(extended_csv_path, index=False, encoding="utf-8")
print("Saved extended CSV (closest +1):", extended_csv_path)

# ---------------------------
# 7) Two figures
# ---------------------------
# Figure 1: depth distribution (DIRECT only) using depth_adjusted (= depth_wn)
direct_depths = direct_table["depth_adjusted"].tolist() if len(direct_table) > 0 else []

plt.figure()
plt.hist(direct_depths, bins=15)
plt.xlabel("Depth in WordNet hierarchy")
plt.ylabel("Number of synsets (question_type = O)")
plt.title("Depth distribution of direct mapping in WordNet")
plt.show()

# Figure 2: depth distribution (EXTENDED) using depth_adjusted (closest = depth_wn + 1)
extended_depths = extended_table["depth_adjusted"].tolist() if len(extended_table) > 0 else []

plt.figure()
plt.hist(extended_depths, bins=15)
plt.xlabel("Depth in WordNet hierarchy")
plt.ylabel("Number of synsets (question_type = O)")
plt.title("Depth distribution for both direct and second level mapping in WordNet")
plt.show()

# ---------------------------
# 8) Quick stats printout
# ---------------------------
def stats(name, arr):
    if not arr:
        print(name, ": empty")
        return
    print(f"{name}: n={len(arr)}, min={min(arr)}, max={max(arr)}, mean={sum(arr)/len(arr):.2f}")

stats("DIRECT depth_adjusted", direct_depths)
stats("EXTENDED depth_adjusted (closest +1)", extended_depths)