import pandas as pd

# Reload dataset
mapping_path = r"C:\Users\Paola Marongiu\Documents\ALT\LREC\agricoltura-WN - mapping_completo.csv"
df = pd.read_csv(mapping_path, encoding="latin1", sep=None, engine="python")

# Filtra solo tpdom = "O"
df_O = df[df["semanticnet_tpdom"] == "O"]

# =====================
# CHIAVI (concept-level)
# =====================
chiavi_status = df_O.groupby("semanticnet_chiave").agg(
    direct=("semanticnet_chiave_WN_1", lambda x: x.notna().any()),
    closest=("semanticnet_chiave_WN_closest", lambda x: x.notna().any())
).reset_index()

chiavi_total = chiavi_status["semanticnet_chiave"].nunique()
chiavi_both = (chiavi_status["direct"] & chiavi_status["closest"]).sum()
chiavi_direct_only = (chiavi_status["direct"] & ~chiavi_status["closest"]).sum()
chiavi_closest_only = (~chiavi_status["direct"] & chiavi_status["closest"]).sum()
chiavi_any_mapped = (chiavi_status["direct"] | chiavi_status["closest"]).sum()
chiavi_fully_unmapped = (~chiavi_status["direct"] & ~chiavi_status["closest"]).sum()

# =====================
# PAROLE (word-level)
# =====================
parole_status = df_O.groupby("semanticnet_parola").agg(
    direct=("semanticnet_parola_WN", lambda x: x.notna().any()),
    closest=("semanticnet_parola_WN_closest", lambda x: x.notna().any())
).reset_index()

parole_total = parole_status["semanticnet_parola"].nunique()
parole_both = (parole_status["direct"] & parole_status["closest"]).sum()
parole_direct_only = (parole_status["direct"] & ~parole_status["closest"]).sum()
parole_closest_only = (~parole_status["direct"] & parole_status["closest"]).sum()
parole_any_mapped = (parole_status["direct"] | parole_status["closest"]).sum()
parole_fully_unmapped = (~parole_status["direct"] & ~parole_status["closest"]).sum()

# =====================
# TABELLA RIASSUNTIVA (2 RIGHE)
# =====================
summary_df = pd.DataFrame([
    {
        "Level": "Chiavi (tpdom=O)",
        "Total": chiavi_total,
        "Direct_only": int(chiavi_direct_only),
        "Closest_only": int(chiavi_closest_only),
        "Both": int(chiavi_both),
        "Any_mapped": int(chiavi_any_mapped),
        "Fully_unmapped": int(chiavi_fully_unmapped),
    },
    {
        "Level": "Parole (tpdom=O)",
        "Total": parole_total,
        "Direct_only": int(parole_direct_only),
        "Closest_only": int(parole_closest_only),
        "Both": int(parole_both),
        "Any_mapped": int(parole_any_mapped),
        "Fully_unmapped": int(parole_fully_unmapped),
    }
])

# =====================
# STAMPA LISTA PAROLE (tpdom = "O")
# =====================

lista_parole = sorted(df_O["semanticnet_parola"].dropna().unique())

print("\nLista delle parole (tpdom = 'O'):")
print("Totale parole:", len(lista_parole))
print("-" * 40)

for parola in lista_parole:
    print(parola)

# Esporta CSV
output_path = r"C:\Users\Paola Marongiu\Documents\ALT\LREC\summary_tpdom_O_breakdown.csv"
summary_df.to_csv(output_path, index=False, encoding="utf-8")

print("Tabella salvata in:", output_path)
summary_df