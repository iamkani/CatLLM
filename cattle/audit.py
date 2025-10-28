import pandas as pd

REQUIRED_COLS = ["member_id","herd_id","breed","birth_weight","weaning_weight","yearling_weight","genomic_test"]

def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    # Return rows with any missing required fields
    cols = [c for c in REQUIRED_COLS if c in df.columns]
    mask = df[cols].isna() | (df[cols].astype(str).str.strip().eq(""))
    need = mask.any(axis=1)
    out = df.loc[need, ["member_id"] + cols].copy()
    out["missing_fields"] = mask.loc[need].apply(lambda r: [c for c,v in r.items() if v], axis=1)
    return out
