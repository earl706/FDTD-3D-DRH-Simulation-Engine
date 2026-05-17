"""
Anatomy descriptor: tissue tables, visualization config, and target/healthy label sets.

Single structure consumed by the shared pipeline (run_simulation) for brain, breast, cervix.
"""

from dataclasses import dataclass

from hermes_drh.settings.tissues import get_tissue_tables
from hermes_drh.settings.visualization import VisualizationConfig, get_visualization_config


@dataclass
class AnatomyConfig:
    """Full config for one anatomy: materials, viz, and target/healthy label sets."""

    name: str
    tissue_table: dict
    k_tissue: dict
    visualization_config: VisualizationConfig
    target_labels: list
    healthy_labels: list


def get_anatomy_config(anatomy: str) -> AnatomyConfig:
    """
    Return AnatomyConfig for the given anatomy.

    Fills tissue_table, k_tissue, visualization_config from config modules and
    sets target_labels (e.g. tumor) and healthy_labels (e.g. normal tissue).
    """
    a = anatomy.lower().strip()
    tissue_table, k_tissue = get_tissue_tables(a)
    viz_config = get_visualization_config(a)
    if a == "brain":
        return AnatomyConfig(
            name="brain",
            tissue_table=tissue_table,
            k_tissue=k_tissue,
            visualization_config=viz_config,
            target_labels=[1, 2, 3],
            healthy_labels=[4],
        )
    if a == "breast":
        return AnatomyConfig(
            name="breast",
            tissue_table=tissue_table,
            k_tissue=k_tissue,
            visualization_config=viz_config,
            target_labels=[1],
            healthy_labels=[2],
        )
    if a == "cervix":
        return AnatomyConfig(
            name="cervix",
            tissue_table=tissue_table,
            k_tissue=k_tissue,
            visualization_config=viz_config,
            target_labels=[1],
            healthy_labels=[2],
        )
    raise ValueError(
        f"Unknown anatomy: {anatomy!r}. Use 'brain', 'breast', or 'cervix'."
    )
