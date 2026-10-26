"""Small, thesis-ready plot set for one RT evaluation."""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
# Type-3 glyphs produced by Matplotlib's historical default are valid PDF but
# are not rendered by several IDE/remote PDF previews. Embed TrueType fonts so
# the exported figures work in those viewers as well as in browser/PDF readers.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _save(fig, path: Path):
    fig.tight_layout(); fig.savefig(path.with_suffix(".png"), dpi=220); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)

def prediction_plots(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True); obs=frame.observed_rt.to_numpy(); pred=frame.predicted_rt.to_numpy(); err=pred-obs
    fig, ax=plt.subplots(figsize=(5.4,5)); h=ax.hexbin(obs,pred,gridsize=55,mincnt=1,cmap="viridis"); lo,hi=min(obs.min(),pred.min()),max(obs.max(),pred.max()); ax.plot([lo,hi],[lo,hi],"--",color="black",lw=.9); ax.set(xlabel="Observed RT",ylabel="Predicted RT"); fig.colorbar(h,ax=ax,label="Peptides"); _save(fig,output/"observed_vs_predicted")
    for values, name, xlabel in [(err,"residual_distribution","Residual (predicted − observed RT)"),(np.abs(err),"absolute_error_distribution","Absolute error")]:
        fig,ax=plt.subplots(figsize=(6,4)); ax.hist(values,bins=min(60,max(15,len(values)//10)),color="#2878b5",alpha=.9); ax.axvline(0,color="black",lw=.8) if name.startswith("residual") else None; ax.set(xlabel=xlabel,ylabel="Peptides"); _save(fig,output/name)
    fig,ax=plt.subplots(figsize=(6,4)); ax.hexbin(obs,err,gridsize=50,mincnt=1,cmap="viridis"); ax.axhline(0,color="black",lw=.8); ax.set(xlabel="Observed RT",ylabel="Residual (predicted − observed RT)"); _save(fig,output/"error_vs_observed_rt")
    fig,ax=plt.subplots(figsize=(6,4)); groups=[g.absolute_error.values for _,g in frame.groupby("peptide_length")]; labels=[str(k) for k,_ in frame.groupby("peptide_length")]; ax.boxplot(groups,labels=labels,showfliers=False); ax.set(xlabel="Peptide length",ylabel="Absolute error"); _save(fig,output/"absolute_error_by_length")
    for column in ("semi_specific", "dimethyl_class"):
        if column in frame and frame[column].notna().any() and frame[column].nunique() <= 12:
            fig,ax=plt.subplots(figsize=(max(5,frame[column].nunique()*1.1),4)); frame.boxplot(column="absolute_error",by=column,ax=ax,showfliers=False); ax.set(title="",xlabel=column.replace("_"," "),ylabel="Absolute error"); fig.suptitle(""); _save(fig,output/f"absolute_error_by_{column}")

def attribution_plots(attr: pd.DataFrame, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    summary=attr.groupby("relative_position",as_index=False).agg(mean_absolute_attribution=("absolute_attribution","mean"),mean_signed_attribution=("signed_attribution","mean"),n=("peptide_id","size")); summary.to_parquet(output.parent/"attribution_position_summary.parquet",index=False)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(summary.relative_position,summary.mean_absolute_attribution,color="#d95f02",label="absolute"); ax.plot(summary.relative_position,summary.mean_signed_attribution,color="#1b9e77",label="signed"); ax.axhline(0,color="black",lw=.7); ax.set(xlabel="Relative peptide position (0=N-term, 1=C-term)",ylabel="Mean attribution"); ax.legend(frameon=False); _save(fig,output/"attribution_by_relative_position")
    # Predictor tokens, rather than ``residue``, are the scientific unit here.
    # A residue is intentionally None for multi-character modification tokens
    # (e.g. K(Dimethyl:...)); grouping on residue therefore hid precisely the
    # fine-tuned modification embeddings that we want to inspect.
    token=(attr.groupby("predictor_token_string",as_index=False)
           .agg(mean_absolute_attribution=("absolute_attribution","mean"),
                mean_signed_attribution=("signed_attribution","mean"),
                n=("peptide_id","size"))
           .sort_values("mean_absolute_attribution",ascending=False))
    token["is_dimethyl_token"] = token.predictor_token_string.astype(str).str.contains("Dimethyl", case=False, na=False)
    token.to_parquet(output.parent/"attribution_token_summary.parquet",index=False)
    shown=token.head(25).sort_values("mean_absolute_attribution")
    colors=np.where(shown.is_dimethyl_token,"#d95f02","#7570b3")
    fig,ax=plt.subplots(figsize=(9,6)); ax.barh(shown.predictor_token_string.astype(str),shown.mean_absolute_attribution,color=colors); ax.set(xlabel="Mean |Integrated Gradients attribution|",ylabel="DIA-NN predictor token",title="Token contribution to predicted RT"); _save(fig,output/"attribution_by_token")

    # A direct heavy-dimethyl view.  Absolute attribution measures contribution
    # magnitude; signed attribution gives its direction relative to the stated
    # boundary-only reference.  These values are model explanations, not a
    # causal estimate of the chemical modification's RT effect.
    dimethyl=token[token.is_dimethyl_token].sort_values("mean_absolute_attribution",ascending=True)
    dimethyl.to_parquet(output.parent/"dimethyl_token_attribution_summary.parquet",index=False)
    if not dimethyl.empty:
        fig,ax=plt.subplots(figsize=(8,3.6)); ax.barh(dimethyl.predictor_token_string.astype(str),dimethyl.mean_absolute_attribution,color="#d95f02"); ax.set(xlabel="Mean |Integrated Gradients attribution|",ylabel="Heavy-dimethyl predictor token",title="Heavy-dimethyl token contribution to predicted RT"); _save(fig,output/"dimethyl_token_attribution")

    variable=attr[~attr.predictor_token_string.isin(["<", ">"])].copy()
    variable["is_dimethyl_token"] = variable.predictor_token_string.astype(str).str.contains("Dimethyl",case=False,na=False)
    peptide=(variable.groupby(["peptide_id","dimethyl_class"],dropna=False,as_index=False)
             .agg(total_absolute_attribution=("absolute_attribution","sum"),
                  dimethyl_absolute_attribution=("absolute_attribution",lambda values: values[variable.loc[values.index,"is_dimethyl_token"]].sum()),
                  dimethyl_signed_attribution=("signed_attribution",lambda values: values[variable.loc[values.index,"is_dimethyl_token"]].sum()),
                  n_dimethyl_tokens=("is_dimethyl_token","sum")))
    peptide["dimethyl_attribution_share"] = np.divide(peptide.dimethyl_absolute_attribution, peptide.total_absolute_attribution, out=np.zeros(len(peptide)), where=peptide.total_absolute_attribution.to_numpy()!=0)
    peptide.to_parquet(output.parent/"dimethyl_peptide_attribution_summary.parquet",index=False)
    classes=[x for x in ["normal","heavy_nterm","heavy_lysine","heavy_both"] if x in set(peptide.dimethyl_class)]
    if classes:
        values=[peptide.loc[peptide.dimethyl_class.eq(name),"dimethyl_attribution_share"] for name in classes]
        fig,ax=plt.subplots(figsize=(8,4)); ax.boxplot(values,tick_labels=classes,showfliers=False); ax.set(ylabel="Share of total |attribution| assigned to dimethyl tokens",xlabel="Peptide dimethyl class",title="Relative model reliance on dimethyl tokens"); ax.set_ylim(bottom=0); _save(fig,output/"dimethyl_attribution_share_by_peptide_class")
    return summary
