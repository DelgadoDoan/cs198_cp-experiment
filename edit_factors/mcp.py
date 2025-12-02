import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import make_ideal_cp, edit_pp, edit_cc, edit_cp

# ---------------------------------------------------
# Main program
# ---------------------------------------------------
mcp_pairs = [
    [(20, 20), (20, 20)],
    [(50, 50), (50, 50)],
    [(100, 100), (100, 100)],
    [(20, 50), (20, 50)],
    [(20, 100), (20, 100)],
    [(50, 20), (50, 20)],
    [(100, 20), (100, 20)],
    [(20, 50), (50, 20)],
    [(20, 100), (100, 20)],
]

edit_factors = [round(x * 0.05, 2) for x in range(11)]
print("edit factors:", edit_factors)

for cp_pair in mcp_pairs:
    tol = 1e-2

    data_cc = []
    data_pp = []
    data_cp = []

    # Create 3 subplots: one for each edit type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- 1. Edited CC ---
    for edit_factor in edit_factors:
        cp = make_ideal_cp(cp_pair)
        edited_cc = edit_cc(cp, cp_pair, edit_factor)
        eigvals_cc, eigvecs_cc = np.linalg.eig(edited_cc)
        axes[0].scatter(
            [edit_factor]*len(eigvals_cc),
            np.real(eigvals_cc),
            facecolors='none',
            edgecolors='red',
            alpha=0.6,
            s=5)
        for eigval in eigvals_cc:
            data_cc.append({
                "Edit Factor": edit_factor,
                "Eigenvalue (Real)": np.real(eigval),
                "Eigenvalue (Imag)": np.imag(eigval)
            })

    axes[0].set_title("Edited CC")
    axes[0].set_xlabel("Edit Factor")
    axes[0].set_ylabel("Eigenvalue (Real Part)")
    axes[0].grid(True)

    # --- 2. Edited PP ---
    for edit_factor in edit_factors:
        cp = make_ideal_cp(cp_pair)
        edited_pp = edit_pp(cp, cp_pair, edit_factor)
        eigvals_pp, eigvecs_pp = np.linalg.eig(edited_pp)
        axes[1].scatter(
            [edit_factor]*len(eigvals_pp),
            np.real(eigvals_pp),
            facecolors='none',
            edgecolors='blue',
            alpha=0.6,
            s=5)
        for eigval in eigvals_pp:
            data_pp.append({
                "Edit Factor": edit_factor,
                "Eigenvalue (Real)": np.real(eigval),
                "Eigenvalue (Imag)": np.imag(eigval)
            })
        
    axes[1].set_title("Edited PP")
    axes[1].set_xlabel("Edit Factor")
    axes[1].grid(True)

    # --- 3. Edited CP ---
    for edit_factor in edit_factors:
        cp = make_ideal_cp(cp_pair)
        edited_cp = edit_cp(cp, edit_factor)
        eigvals_cp, eigvecs_cp = np.linalg.eig(edited_cp)
        axes[2].scatter(
            [edit_factor]*len(eigvals_cp),
            np.real(eigvals_cp),
            facecolors='none',
            edgecolors='green',
            alpha=0.6,
            s=5)
        for eigval in eigvals_cp:
            data_cp.append({
                "Edit Factor": edit_factor,
                "Eigenvalue (Real)": np.real(eigval),
                "Eigenvalue (Imag)": np.imag(eigval)
            })
        
    axes[2].set_title("Edited CP")
    axes[2].set_xlabel("Edit Factor")
    axes[2].grid(True)

    plt.suptitle("Real Eigenvalues vs Edit Factor for Each Edit Type", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot
    save_dir = "edit_factors/mcp/plots/eigenvalues"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{cp_pair}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Save CSVs ---
    data_dir = f"edit_factors/mcp/data/{cp_pair}"
    os.makedirs(data_dir, exist_ok=True)

    df_cc = pd.DataFrame(data_cc)
    df_pp = pd.DataFrame(data_pp)
    df_cp = pd.DataFrame(data_cp)

    csv_path_cc = os.path.join(data_dir, f"cc_edits.csv")
    csv_path_pp = os.path.join(data_dir, f"pp_edits.csv")
    csv_path_cp = os.path.join(data_dir, f"cp_edits.csv")

    df_cc.to_csv(csv_path_cc, index=False)
    df_pp.to_csv(csv_path_pp, index=False)
    df_cp.to_csv(csv_path_cp, index=False)

    # --- Focused Subplots for Eigenvalues near 0 and -1 ---
    colors = {"CC": "red", "PP": "blue", "CP": "green"}

    # Convert to DataFrames
    df_cc = pd.DataFrame(data_cc)
    df_pp = pd.DataFrame(data_pp)
    df_cp = pd.DataFrame(data_cp)

    # Define focus ranges
    focus_specs = [
        {"target": 0,  "ymin": -0.25, "ymax": 0.25,  "title": "Eigenvalues near 0"},
        {"target": -1, "ymin": -1.25, "ymax": -0.75, "title": "Eigenvalues near -1"}
    ]

    # Prepare figure: 2 rows (targets) × 3 columns (CC, PP, CP)
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), sharex=True)

    for row, spec in enumerate(focus_specs):
        target = spec["target"]
        ymin, ymax = spec["ymin"], spec["ymax"]

        for col, (df, label) in enumerate([(df_cc, "CC"), (df_pp, "PP"), (df_cp, "CP")]):
            ax = axes[row, col]

            # Filter eigenvalues within the desired vertical range
            df_focus = df[(df["Eigenvalue (Real)"] >= ymin) &
                        (df["Eigenvalue (Real)"] <= ymax)]

            # Scatter plot
            ax.scatter(
                df_focus["Edit Factor"], df_focus["Eigenvalue (Real)"],
                facecolors='none',
                edgecolors=colors[label],
                alpha=0.7,
                s=10,
                label=f"{label} eigenvalues"
            )

            # Draw tolerance band as a filled area
            ax.fill_between(
                df_focus["Edit Factor"].unique(),
                target - tol,
                target + tol,
                color='gray',
                alpha=0.2,
                label='Tolerance Band'
            )

            # Central (ideal) bound line
            ax.axhline(y=target, color='black', linestyle='-', linewidth=1, alpha=0.8)

            # Labels and limits
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("Edit Factor")
            ax.set_title(f"{spec['title']} ({label})")
            ax.grid(True, linestyle=':')
            ax.legend(loc="upper right", fontsize=8)

    # Shared Y labels
    axes[0, 0].set_ylabel("Eigenvalue (Real Part)")
    axes[1, 0].set_ylabel("Eigenvalue (Real Part)")

    plt.suptitle("Focused Eigenvalue Ranges with Tolerance Bands", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the combined focused plot
    focus_dir = "edit_factors/mcp/plots/small"
    os.makedirs(focus_dir, exist_ok=True)
    focus_path = os.path.join(focus_dir, f"{cp_pair}.png")
    plt.savefig(focus_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

    # --- Unified bar chart for counts vs edit factors ---
    counts_dir = "edit_factors/mcp/plots/counts"
    os.makedirs(counts_dir, exist_ok=True)

    # Count eigenvalues near 0 and -1 for each edit factor
    def count_eigvals(df, target, tol):
        return [((df[df["Edit Factor"]==ef]["Eigenvalue (Real)"] >= target-tol) &
                (df[df["Edit Factor"]==ef]["Eigenvalue (Real)"] <= target+tol)).sum()
                for ef in edit_factors]

    counts = {
        "CC_0": count_eigvals(df_cc, 0, tol),
        "CC_-1": count_eigvals(df_cc, -1, tol),
        "PP_0": count_eigvals(df_pp, 0, tol),
        "PP_-1": count_eigvals(df_pp, -1, tol),
        "CP_0": count_eigvals(df_cp, 0, tol),
        "CP_-1": count_eigvals(df_cp, -1, tol),
    }

    # Create bar chart: 2 rows (CC, PP) × 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    x = np.arange(len(edit_factors))
    width = 0.35

    # CC
    axes[0].bar(x - width/2, counts["CC_0"], width, label='0 Eigenvalues', color='red', alpha=0.7)
    axes[0].bar(x + width/2, counts["CC_-1"], width, label='-1 Eigenvalues', color='orange', alpha=0.7)
    axes[0].set_ylabel("Count")
    axes[0].set_title("CC: Eigenvalue Counts Near 0 and -1")
    axes[0].grid(True, linestyle=':')
    axes[0].legend()

    # PP
    axes[1].bar(x - width/2, counts["PP_0"], width, label='0 Eigenvalues', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, counts["PP_-1"], width, label='-1 Eigenvalues', color='cyan', alpha=0.7)
    axes[1].set_xlabel("Edit Factor")
    axes[1].set_ylabel("Count")
    axes[1].set_title("PP: Eigenvalue Counts Near 0 and -1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(edit_factors)
    axes[1].grid(True, linestyle=':')
    axes[1].legend()

    # CP
    axes[2].bar(x - width/2, counts["CP_0"], width, label='0 Eigenvalues', color='blue', alpha=0.7)
    axes[2].bar(x + width/2, counts["CP_-1"], width, label='-1 Eigenvalues', color='cyan', alpha=0.7)
    axes[2].set_xlabel("Edit Factor")
    axes[2].set_ylabel("Count")
    axes[2].set_title("CP: Eigenvalue Counts Near 0 and -1")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(edit_factors)
    axes[2].grid(True, linestyle=':')
    axes[2].legend()

    plt.suptitle("Counts of Eigenvalues Near 0 and -1 vs Edit Factor", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the bar chart
    bar_path = os.path.join(counts_dir, f"{cp_pair}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Eigenvector plots (Principal + 2nd Smallest) ---
    fig_u, axes_u = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    datasets = [
        ("CC", eigvals_cc, eigvecs_cc, "red"),
        ("PP", eigvals_pp, eigvecs_pp, "blue"),
        ("CP", eigvals_cp, eigvecs_cp, "green")
    ]

    for ax, (label, eigvals, eigvecs, color) in zip(axes_u, datasets):
        # Sort eigenvalues
        idx_sorted = np.argsort(eigvals)
        smallest_idx = idx_sorted[0]
        second_smallest_idx = idx_sorted[1]
        principal_idx = idx_sorted[-1]

        x = np.arange(eigvecs.shape[0])

        # Principal eigenvector
        ax.plot(
            x, np.real(eigvecs[:, principal_idx]),
            color=color, linewidth=1.2, label="Principal Eigenvector"
        )

        # 2nd smallest eigenvector
        ax.plot(
            x, np.real(eigvecs[:, second_smallest_idx]),
            color=color, linestyle="--", linewidth=1.2, label="2nd Smallest Eigenvector"
        )

        ax.set_title(f"{label} (edit={edit_factor})")
        ax.set_xlabel("Node Index")
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=8)

    axes_u[0].set_ylabel("Eigenvector Component")
    plt.tight_layout()

    # Save plot
    eigenvec_dir = "edit_factors/mcp/plots/eigenvectors"
    os.makedirs(eigenvec_dir, exist_ok=True)
    save_path = os.path.join(eigenvec_dir, f"{cp_pair}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig_u)

    print(f"{cp_pair} done")