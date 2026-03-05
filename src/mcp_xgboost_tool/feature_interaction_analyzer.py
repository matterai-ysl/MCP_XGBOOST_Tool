"""
Feature Interaction Decoupling Module

Decouples pairwise feature interactions to identify WHERE two features
exhibit synergy (cooperative gain) vs antagonism (counteracting effect).

This module is distinct from the SHAP interaction plots in the global
feature importance analysis (feature_importance.py):
  - Global SHAP: raw interaction heatmaps for ALL pairs as part of an
    overall importance ranking — qualitative overview.
  - This module: targeted decoupling of ONE specific feature pair with
    Gaussian kernel smoothing to extract robust synergy/antagonism region
    boundaries and quantified descriptions — quantitative region analysis.

Algorithm:
  1. Compute SHAP interaction values for the specified feature pair
  2. Build a 2D grid and compute bin-averaged interaction values
  3. Apply Gaussian kernel smoothing to obtain a robust interaction surface
     (avoids strict per-sample >0 / <0 judgment, captures macro trends)
  4. Extract the zero-level contour as the synergy/antagonism boundary
  5. Label connected regions and report their value ranges and intensities
"""

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter, label as ndimage_label

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. Feature interaction analysis will be disabled.")

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")


class FeatureInteractionAnalyzer:
    """
    Decouples pairwise feature interactions to identify synergistic and
    antagonistic regions between two specified features.

    Unlike the raw SHAP interaction heatmaps produced by the global feature
    importance module, this analyzer applies Gaussian kernel smoothing and
    contour-based boundary extraction to produce robust region classifications
    that are not overly sensitive to per-sample noise — capturing the macro
    trend of where two features cooperate vs counteract.
    """

    def __init__(self, output_dir: str = "feature_interaction_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_interaction(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        feature_1: str,
        feature_2: str,
        X_display: Optional[np.ndarray] = None,
        grid_resolution: int = 30,
        smoothing_sigma: Optional[float] = None,
        max_samples: int = 2000,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze the interaction between two features.

        Args:
            model: Trained XGBoost model.
            X: Processed feature matrix (n_samples, n_features) used for SHAP.
            feature_names: Ordered list of feature names matching columns of X.
            feature_1: Name of the first feature.
            feature_2: Name of the second feature.
            X_display: Raw (unprocessed) feature matrix for axis labels.
                       Falls back to X when not provided.
            grid_resolution: Number of bins along each axis (default 30).
            smoothing_sigma: Gaussian kernel sigma. ``None`` selects an
                adaptive value (grid_resolution / 10).
            max_samples: Max samples fed to the SHAP explainer.
            generate_plots: Whether to produce PNG visualisations.

        Returns:
            Dict with keys: feature_1, feature_2, regions, boundary_contours,
            smoothing_sigma, grid_resolution, plots, raw_interaction_stats, …
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP library is not installed. Run: pip install shap"}

        if feature_1 not in feature_names:
            return {"error": f"Feature '{feature_1}' not found. Available: {feature_names}"}
        if feature_2 not in feature_names:
            return {"error": f"Feature '{feature_2}' not found. Available: {feature_names}"}
        if feature_1 == feature_2:
            return {"error": "feature_1 and feature_2 must be different features"}

        idx1 = feature_names.index(feature_1)
        idx2 = feature_names.index(feature_2)

        if X_display is None:
            X_display = X

        # --- Sampling ------------------------------------------------
        n = X.shape[0]
        if n > max_samples:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n, max_samples, replace=False)
            X_sample = X[sample_idx]
            X_display_sample = X_display[sample_idx]
        else:
            X_sample = X
            X_display_sample = X_display

        logger.info(
            f"Computing SHAP interaction values for ({feature_1}, {feature_2}) "
            f"on {X_sample.shape[0]} samples …"
        )

        # --- SHAP interaction values ----------------------------------
        explainer = shap.TreeExplainer(model)
        shap_interaction = explainer.shap_interaction_values(X_sample)

        if isinstance(shap_interaction, list):
            shap_interaction = shap_interaction[0]

        pair_values = shap_interaction[:, idx1, idx2]  # (n_samples,)
        f1_vals = X_display_sample[:, idx1].astype(float)
        f2_vals = X_display_sample[:, idx2].astype(float)

        # --- Build smoothed 2D surface --------------------------------
        if smoothing_sigma is None:
            smoothing_sigma = max(grid_resolution / 10.0, 1.0)

        surface, xedges, yedges, counts = self._build_smoothed_surface(
            f1_vals, f2_vals, pair_values,
            grid_resolution=grid_resolution,
            sigma=smoothing_sigma,
        )

        # --- Region extraction ----------------------------------------
        regions, label_matrix = self._extract_regions(
            surface, xedges, yedges, counts,
            feature_1=feature_1,
            feature_2=feature_2,
        )

        # --- Raw statistics -------------------------------------------
        raw_stats = {
            "n_samples": int(X_sample.shape[0]),
            "interaction_mean": float(np.mean(pair_values)),
            "interaction_std": float(np.std(pair_values)),
            "interaction_min": float(np.min(pair_values)),
            "interaction_max": float(np.max(pair_values)),
            "positive_ratio": float(np.mean(pair_values > 0)),
            "negative_ratio": float(np.mean(pair_values < 0)),
        }

        # --- Export raw data for user's own plotting ---------------------
        data_files = self._export_raw_data(
            f1_vals, f2_vals, pair_values,
            surface, xedges, yedges, counts, label_matrix,
            feature_1, feature_2, regions,
        )

        # --- Visualisation --------------------------------------------
        plot_paths: List[str] = []
        if generate_plots:
            plot_paths = self._create_plots(
                surface, xedges, yedges, label_matrix, counts,
                f1_vals, f2_vals, pair_values,
                feature_1, feature_2,
                regions, smoothing_sigma,
            )

        # --- HTML report -------------------------------------------------
        report_path = self._generate_html_report(
            feature_1, feature_2, regions, raw_stats,
            smoothing_sigma, grid_resolution, plot_paths,
        )

        result: Dict[str, Any] = {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "regions": regions,
            "raw_interaction_stats": raw_stats,
            "smoothing_sigma": smoothing_sigma,
            "grid_resolution": grid_resolution,
            "plots": plot_paths,
            "report_path": report_path,
            "data_files": data_files,
        }

        logger.info(
            f"Interaction analysis complete: {len(regions)} regions detected "
            f"({sum(1 for r in regions if r['type'] == 'synergy')} synergy, "
            f"{sum(1 for r in regions if r['type'] == 'antagonism')} antagonism)"
        )
        return result

    # ------------------------------------------------------------------
    # Internal: smoothed surface
    # ------------------------------------------------------------------

    @staticmethod
    def _build_smoothed_surface(
        f1: np.ndarray,
        f2: np.ndarray,
        interaction: np.ndarray,
        grid_resolution: int = 30,
        sigma: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin-average interaction values onto a 2-D grid, then Gaussian-smooth.

        Returns:
            surface: (grid_resolution, grid_resolution) smoothed interaction
            xedges, yedges: bin edge arrays
            counts: raw sample counts per bin (before smoothing)
        """
        f1_min, f1_max = float(np.min(f1)), float(np.max(f1))
        f2_min, f2_max = float(np.min(f2)), float(np.max(f2))
        f1_pad = (f1_max - f1_min) * 0.01 or 0.5
        f2_pad = (f2_max - f2_min) * 0.01 or 0.5

        xedges = np.linspace(f1_min - f1_pad, f1_max + f1_pad, grid_resolution + 1)
        yedges = np.linspace(f2_min - f2_pad, f2_max + f2_pad, grid_resolution + 1)

        weighted_sum, _, _ = np.histogram2d(f1, f2, bins=[xedges, yedges], weights=interaction)
        counts, _, _ = np.histogram2d(f1, f2, bins=[xedges, yedges])

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_mean = np.where(counts > 0, weighted_sum / counts, 0.0)

        surface = gaussian_filter(raw_mean, sigma=sigma)
        return surface, xedges, yedges, counts

    # ------------------------------------------------------------------
    # Internal: region extraction
    # ------------------------------------------------------------------

    def _extract_regions(
        self,
        surface: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        counts: np.ndarray,
        feature_1: str,
        feature_2: str,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Label contiguous synergy / antagonism regions on the smoothed surface.

        Returns a list of region descriptors and the integer label matrix.
        """
        sign_matrix = np.sign(surface)

        pos_mask = (sign_matrix > 0).astype(int)
        neg_mask = (sign_matrix < 0).astype(int)

        pos_labels, n_pos = ndimage_label(pos_mask)
        neg_labels, n_neg = ndimage_label(neg_mask)

        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        total_data_cells = int(np.sum(counts > 0))
        regions: List[Dict[str, Any]] = []

        combined_labels = np.zeros_like(surface, dtype=int)
        current_label = 0

        for region_id in range(1, n_pos + 1):
            mask = pos_labels == region_id
            region_cells_with_data = int(np.sum((counts > 0) & mask))
            if region_cells_with_data == 0:
                continue

            current_label += 1
            combined_labels[mask] = current_label

            rows, cols = np.where(mask)
            f1_range = [float(xcenters[rows.min()]), float(xcenters[rows.max()])]
            f2_range = [float(ycenters[cols.min()]), float(ycenters[cols.max()])]
            avg_val = float(np.mean(surface[mask]))
            area_pct = (
                round(100.0 * region_cells_with_data / total_data_cells, 1)
                if total_data_cells > 0
                else 0.0
            )

            regions.append(
                {
                    "type": "synergy",
                    "label_id": current_label,
                    "feature_1_range": [round(f1_range[0], 4), round(f1_range[1], 4)],
                    "feature_2_range": [round(f2_range[0], 4), round(f2_range[1], 4)],
                    "average_interaction_value": round(avg_val, 6),
                    "data_coverage_percent": area_pct,
                    "description": (
                        f"Synergy: when {feature_1} ∈ [{f1_range[0]:.4g}, {f1_range[1]:.4g}] "
                        f"and {feature_2} ∈ [{f2_range[0]:.4g}, {f2_range[1]:.4g}], "
                        f"these two features cooperate to increase the prediction "
                        f"(avg interaction = {avg_val:+.4f})"
                    ),
                }
            )

        for region_id in range(1, n_neg + 1):
            mask = neg_labels == region_id
            region_cells_with_data = int(np.sum((counts > 0) & mask))
            if region_cells_with_data == 0:
                continue

            current_label += 1
            combined_labels[mask] = current_label

            rows, cols = np.where(mask)
            f1_range = [float(xcenters[rows.min()]), float(xcenters[rows.max()])]
            f2_range = [float(ycenters[cols.min()]), float(ycenters[cols.max()])]
            avg_val = float(np.mean(surface[mask]))
            area_pct = (
                round(100.0 * region_cells_with_data / total_data_cells, 1)
                if total_data_cells > 0
                else 0.0
            )

            regions.append(
                {
                    "type": "antagonism",
                    "label_id": current_label,
                    "feature_1_range": [round(f1_range[0], 4), round(f1_range[1], 4)],
                    "feature_2_range": [round(f2_range[0], 4), round(f2_range[1], 4)],
                    "average_interaction_value": round(avg_val, 6),
                    "data_coverage_percent": area_pct,
                    "description": (
                        f"Antagonism: when {feature_1} ∈ [{f1_range[0]:.4g}, {f1_range[1]:.4g}] "
                        f"and {feature_2} ∈ [{f2_range[0]:.4g}, {f2_range[1]:.4g}], "
                        f"these two features counteract each other, reducing the prediction "
                        f"(avg interaction = {avg_val:+.4f})"
                    ),
                }
            )

        regions.sort(key=lambda r: abs(r["average_interaction_value"]), reverse=True)
        return regions, combined_labels

    # ------------------------------------------------------------------
    # Internal: visualisation
    # ------------------------------------------------------------------

    def _create_plots(
        self,
        surface: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        label_matrix: np.ndarray,
        counts: np.ndarray,
        f1_vals: np.ndarray,
        f2_vals: np.ndarray,
        pair_values: np.ndarray,
        feature_1: str,
        feature_2: str,
        regions: List[Dict[str, Any]],
        sigma: float,
    ) -> List[str]:
        """Generate publication-quality visualisations."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_f1 = feature_1.replace("/", "_").replace(" ", "_")
        safe_f2 = feature_2.replace("/", "_").replace(" ", "_")
        paths: List[str] = []

        paths.append(
            self._plot_smoothed_heatmap_with_boundary(
                surface, xedges, yedges, counts,
                feature_1, feature_2, sigma, ts, safe_f1, safe_f2,
            )
        )

        paths.append(
            self._plot_region_map(
                surface, xedges, yedges, label_matrix, counts, regions,
                feature_1, feature_2, ts, safe_f1, safe_f2,
            )
        )

        paths.append(
            self._plot_scatter_with_boundary(
                surface, xedges, yedges,
                f1_vals, f2_vals, pair_values,
                feature_1, feature_2, ts, safe_f1, safe_f2,
            )
        )

        return paths

    # -- Plot 1: smoothed heatmap + zero contour -----------------------

    def _plot_smoothed_heatmap_with_boundary(
        self, surface, xedges, yedges, counts,
        f1, f2, sigma, ts, sf1, sf2,
    ) -> str:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
        plt.rcParams["font.family"] = "Arial"

        masked = np.where(counts > 0, surface, np.nan)

        finite = masked[np.isfinite(masked)]
        if len(finite) > 0:
            abs_max = max(abs(np.nanmin(finite)), abs(np.nanmax(finite))) or 0.5
        else:
            abs_max = 0.5

        cmap = self._synergy_antagonism_cmap()
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        im = ax.imshow(
            masked.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap, norm=norm, interpolation="bilinear",
        )

        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        X_grid, Y_grid = np.meshgrid(xc, yc)
        ax.contour(
            X_grid, Y_grid, surface.T,
            levels=[0], colors="black", linewidths=2.5, linestyles="--",
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label("Smoothed SHAP interaction value", fontsize=14)
        cbar.ax.tick_params(labelsize=11)

        ax.set_xlabel(f"{f1}", fontsize=16)
        ax.set_ylabel(f"{f2}", fontsize=16)
        ax.set_title(
            f"Feature Interaction: {f1} × {f2}\n"
            f"(Gaussian smoothed, σ={sigma:.1f}  |  --- boundary = 0)",
            fontsize=15, pad=12,
        )
        ax.tick_params(labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

        plt.tight_layout()
        path = str(self.output_dir / f"interaction_heatmap_{sf1}_vs_{sf2}_{ts}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved smoothed heatmap → {path}")
        return path

    # -- Plot 2: gradient region map with coloured hatching --------------

    def _plot_region_map(
        self, surface, xedges, yedges, label_matrix, counts, regions,
        f1, f2, ts, sf1, sf2,
    ) -> str:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
        plt.rcParams["font.family"] = "Arial"

        masked = np.where(counts > 0, surface, np.nan)

        finite = masked[np.isfinite(masked)]
        abs_max = (
            max(abs(np.nanmin(finite)), abs(np.nanmax(finite)))
            if len(finite) > 0 else 0.5
        ) or 0.5

        cmap = self._synergy_antagonism_cmap()
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        im = ax.imshow(
            masked.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap, norm=norm, interpolation="bilinear",
        )

        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        Xg, Yg = np.meshgrid(xc, yc)

        ax.contour(Xg, Yg, surface.T, levels=[0],
                   colors="black", linewidths=2.5, linestyles="-")

        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors

        synergy_cmap = mcolors.ListedColormap(["none", "#EF444440"])
        antag_cmap = mcolors.ListedColormap(["none", "#3B82F640"])

        ax.contourf(
            Xg, Yg, surface.T, levels=[0, surface.max() + 1],
            colors=["#EF444420", "#EF444420"], hatches=["///"],
        )
        ax.contourf(
            Xg, Yg, surface.T, levels=[surface.min() - 1, 0],
            colors=["#3B82F620", "#3B82F620"], hatches=["\\\\\\"],
        )

        legend_patches = [
            mpatches.Patch(facecolor="#EF4444", alpha=0.45,
                           hatch="///", edgecolor="#B91C1C", label="Synergy (+)"),
            mpatches.Patch(facecolor="#3B82F6", alpha=0.45,
                           hatch="\\\\\\", edgecolor="#1D4ED8", label="Antagonism (−)"),
            mpatches.Patch(facecolor="white", edgecolor="black",
                           linewidth=1.5, label="Boundary (0)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=12,
                  framealpha=0.92, edgecolor="#9CA3AF")

        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label("Smoothed interaction intensity", fontsize=14)
        cbar.ax.tick_params(labelsize=11)

        ax.set_xlabel(f"{f1}", fontsize=16)
        ax.set_ylabel(f"{f2}", fontsize=16)
        ax.set_title(
            f"Synergy / Antagonism Region Map: {f1} × {f2}\n"
            f"(gradient intensity  |  ── boundary = 0)",
            fontsize=15, pad=12,
        )
        ax.tick_params(labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

        plt.tight_layout()
        path = str(self.output_dir / f"interaction_region_map_{sf1}_vs_{sf2}_{ts}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved gradient region map → {path}")
        return path

    # -- Plot 3: scatter with boundary overlay -------------------------

    def _plot_scatter_with_boundary(
        self, surface, xedges, yedges,
        f1_vals, f2_vals, pair_values,
        f1, f2, ts, sf1, sf2,
    ) -> str:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
        plt.rcParams["font.family"] = "Arial"

        abs_max = max(abs(pair_values.min()), abs(pair_values.max())) or 0.5
        cmap = self._synergy_antagonism_cmap()
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        sc = ax.scatter(
            f1_vals, f2_vals, c=pair_values,
            cmap=cmap, norm=norm,
            s=120, marker="s", alpha=0.85, edgecolors="gray", linewidths=0.3,
        )

        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        Xg, Yg = np.meshgrid(xc, yc)
        ax.contour(
            Xg, Yg, surface.T, levels=[0],
            colors="black", linewidths=2.5, linestyles="--",
        )

        cbar = fig.colorbar(sc, ax=ax, shrink=0.75)
        cbar.set_label("SHAP interaction value (raw)", fontsize=14)
        cbar.ax.tick_params(labelsize=11)

        ax.set_xlabel(f"{f1}", fontsize=16)
        ax.set_ylabel(f"{f2}", fontsize=16)
        ax.set_title(
            f"Raw SHAP Interaction Scatter: {f1} × {f2}\n(--- smoothed boundary)",
            fontsize=15, pad=12,
        )
        ax.tick_params(labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

        plt.tight_layout()
        path = str(self.output_dir / f"interaction_scatter_{sf1}_vs_{sf2}_{ts}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved scatter plot → {path}")
        return path

    # -- Raw data export -----------------------------------------------

    def _export_raw_data(
        self,
        f1_vals: np.ndarray,
        f2_vals: np.ndarray,
        pair_values: np.ndarray,
        surface: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        counts: np.ndarray,
        label_matrix: np.ndarray,
        feature_1: str,
        feature_2: str,
        regions: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Export raw data and grid data as CSV files so the user can
        reproduce or customise the plots.
        """
        files: Dict[str, str] = {}

        # 1. Per-sample raw data
        sample_df = pd.DataFrame({
            feature_1: f1_vals,
            feature_2: f2_vals,
            "shap_interaction_value": pair_values,
        })
        sample_path = str(self.output_dir / "raw_sample_interaction_data.csv")
        sample_df.to_csv(sample_path, index=False)
        files["raw_sample_data"] = sample_path
        logger.info(f"Exported per-sample data ({len(sample_df)} rows) → {sample_path}")

        # 2. Smoothed grid data
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])

        region_type_map = {}
        for r in regions:
            region_type_map[r["label_id"]] = r["type"]

        grid_rows = []
        for i, x in enumerate(xc):
            for j, y in enumerate(yc):
                label_id = int(label_matrix[i, j])
                grid_rows.append({
                    f"{feature_1}_bin_center": round(float(x), 6),
                    f"{feature_2}_bin_center": round(float(y), 6),
                    "smoothed_interaction": round(float(surface[i, j]), 8),
                    "sample_count_in_bin": int(counts[i, j]),
                    "region_label_id": label_id,
                    "region_type": region_type_map.get(label_id, "none"),
                })

        grid_df = pd.DataFrame(grid_rows)
        grid_path = str(self.output_dir / "smoothed_grid_data.csv")
        grid_df.to_csv(grid_path, index=False)
        files["smoothed_grid_data"] = grid_path
        logger.info(f"Exported grid data ({len(grid_df)} rows) → {grid_path}")

        # 3. Region summary
        region_df = pd.DataFrame(regions)
        region_path = str(self.output_dir / "region_summary.csv")
        region_df.to_csv(region_path, index=False)
        files["region_summary"] = region_path
        logger.info(f"Exported region summary ({len(region_df)} rows) → {region_path}")

        return files

    # -- Colormap helper -----------------------------------------------

    @staticmethod
    def _synergy_antagonism_cmap() -> LinearSegmentedColormap:
        blues = plt.cm.Blues(np.linspace(1, 0.15, 128))  # type: ignore[attr-defined]
        reds = plt.cm.Reds(np.linspace(0.15, 1, 128))    # type: ignore[attr-defined]
        white = np.array([[1, 1, 1, 1]])
        full = np.vstack([blues, white, reds])
        return LinearSegmentedColormap.from_list("synergy_antagonism", full, N=512)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _generate_html_report(
        self,
        feature_1: str,
        feature_2: str,
        regions: List[Dict[str, Any]],
        raw_stats: Dict[str, Any],
        sigma: float,
        grid_resolution: int,
        plot_paths: List[str],
    ) -> str:
        """Generate a self-contained HTML report with embedded images."""
        import base64

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _embed_image(path: str) -> str:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return f"data:image/png;base64,{b64}"
            except Exception:
                return ""

        synergy_regions = [r for r in regions if r["type"] == "synergy"]
        antag_regions = [r for r in regions if r["type"] == "antagonism"]

        region_rows = ""
        for r in regions:
            color = "#FDECEA" if r["type"] == "synergy" else "#E8F0FE"
            badge_color = "#DC2626" if r["type"] == "synergy" else "#2563EB"
            badge_label = "Synergy (+)" if r["type"] == "synergy" else "Antagonism (−)"
            region_rows += f"""
            <tr style="background:{color}">
              <td><span style="display:inline-block;padding:2px 10px;border-radius:12px;
                   color:#fff;background:{badge_color};font-size:13px">{badge_label}</span></td>
              <td>[{r['feature_1_range'][0]:.4g}, {r['feature_1_range'][1]:.4g}]</td>
              <td>[{r['feature_2_range'][0]:.4g}, {r['feature_2_range'][1]:.4g}]</td>
              <td>{r['average_interaction_value']:+.6f}</td>
              <td>{r['data_coverage_percent']}%</td>
            </tr>"""

        images_html = ""
        titles = [
            "Smoothed Interaction Heatmap with Boundary",
            "Synergy / Antagonism Region Map",
            "Raw SHAP Interaction Scatter with Boundary",
        ]
        for i, p in enumerate(plot_paths):
            src = _embed_image(p)
            if src:
                title = titles[i] if i < len(titles) else f"Plot {i+1}"
                images_html += f"""
                <div style="margin-bottom:30px">
                  <h3 style="color:#374151">{title}</h3>
                  <img src="{src}" style="max-width:100%;border:1px solid #E5E7EB;border-radius:8px" />
                </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feature Interaction Decoupling: {feature_1} × {feature_2}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin:0; padding:0; background:#F9FAFB; color:#1F2937 }}
  .container {{ max-width:1100px; margin:0 auto; padding:30px 24px }}
  h1 {{ color:#111827; border-bottom:3px solid #3B82F6; padding-bottom:12px }}
  h2 {{ color:#1E40AF; margin-top:36px }}
  table {{ border-collapse:collapse; width:100%; margin:16px 0 }}
  th, td {{ padding:10px 14px; text-align:left; border:1px solid #D1D5DB }}
  th {{ background:#F3F4F6; font-weight:600 }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin:16px 0 }}
  .stat-card {{ background:#fff; border:1px solid #E5E7EB; border-radius:8px; padding:16px; text-align:center }}
  .stat-value {{ font-size:24px; font-weight:700; color:#1E40AF }}
  .stat-label {{ font-size:13px; color:#6B7280; margin-top:4px }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:12px; font-size:13px; color:#fff }}
  .method-note {{ background:#FEF3C7; border-left:4px solid #F59E0B; padding:12px 16px; border-radius:4px; margin:16px 0; font-size:14px }}
</style>
</head>
<body>
<div class="container">

<h1>Feature Interaction Decoupling Report</h1>
<p style="color:#6B7280;font-size:14px">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<h2>1. Analysis Target</h2>
<table>
  <tr><th>Feature 1</th><td><strong>{feature_1}</strong></td></tr>
  <tr><th>Feature 2</th><td><strong>{feature_2}</strong></td></tr>
</table>

<div class="method-note">
  <strong>Method:</strong> SHAP interaction values → 2-D binned averaging (grid {grid_resolution}×{grid_resolution})
  → Gaussian kernel smoothing (σ = {sigma:.2f}) → zero-contour boundary extraction → connected-region labeling.<br>
  <strong>Difference from global SHAP interaction heatmaps:</strong> Global SHAP produces raw (un-smoothed)
  interaction heatmaps for ALL feature pairs as a qualitative overview. This analysis performs targeted
  decoupling of this specific pair with smoothing to extract robust synergy / antagonism region boundaries.
</div>

<h2>2. Raw SHAP Interaction Statistics</h2>
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">{raw_stats['n_samples']}</div>
    <div class="stat-label">Samples analysed</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{raw_stats['interaction_mean']:+.4f}</div>
    <div class="stat-label">Mean interaction</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{raw_stats['interaction_std']:.4f}</div>
    <div class="stat-label">Std deviation</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{raw_stats['interaction_min']:+.4f}</div>
    <div class="stat-label">Min interaction</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{raw_stats['interaction_max']:+.4f}</div>
    <div class="stat-label">Max interaction</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{raw_stats['positive_ratio']:.1%}</div>
    <div class="stat-label">Synergy ratio (raw &gt; 0)</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{raw_stats['negative_ratio']:.1%}</div>
    <div class="stat-label">Antagonism ratio (raw &lt; 0)</div>
  </div>
</div>

<h2>3. Detected Regions (sorted by |interaction|)</h2>
<p>Total: <strong>{len(synergy_regions)}</strong> synergy region(s),
   <strong>{len(antag_regions)}</strong> antagonism region(s).</p>
<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>{feature_1} range</th>
      <th>{feature_2} range</th>
      <th>Avg interaction</th>
      <th>Data coverage</th>
    </tr>
  </thead>
  <tbody>{region_rows}</tbody>
</table>

<h2>4. Region Descriptions</h2>
<ul>
{"".join(f'<li style="margin-bottom:8px">{r["description"]}</li>' for r in regions)}
</ul>

<h2>5. Visualisations</h2>
{images_html}

</div>
</body>
</html>"""

        safe_f1 = feature_1.replace("/", "_").replace(" ", "_")
        safe_f2 = feature_2.replace("/", "_").replace(" ", "_")
        report_file = str(
            self.output_dir / f"interaction_decoupling_report_{safe_f1}_vs_{safe_f2}_{ts}.html"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report saved → {report_file}")
        return report_file
