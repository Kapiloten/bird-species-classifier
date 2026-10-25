
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix,precision_recall_curve, average_precision_score, precision_score
import matplotlib.pyplot as plt


from scripts.project_paths import RAW_DATA_DIR, project_path
from scripts.multilabel_data import labels_for_windows, recording_groups, split_grouped_data
from scripts.single_label import increase_rare_birds as increase_rare_birds


def species_vocabulary(sample_submission=None):
    """Read the official target order from the project-root-relative metadata."""
    path = project_path(sample_submission) if sample_submission is not None else RAW_DATA_DIR / "sample_submission.csv"
    return [str(col).strip() for col in pd.read_csv(path, nrows=0).columns if col != "row_id"]


def index_to_labels(index):
    df_sub = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
    Y = np.load(project_path("ressource/processed_data4/y_labels_train_soundscapes.npy"))

    species = [col for col in df_sub.columns if col != 'row_id']
    vect = Y[index]
    indexes = np.where(vect == 1)[0]

    for index in indexes:
        print(f"{species[index]}")


def createLabelsWithCsv(processedData, csvFile, output_path="ressource/processed_data4/y_labels_train_soundscapes.npy", sample_submission=None):
    """Align annotation targets to extracted row IDs and save them locally."""
    rows = np.load(project_path(processedData))
    targets = labels_for_windows(rows, pd.read_csv(project_path(csvFile)), species_vocabulary(sample_submission))
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, targets)
    return targets


def partition_data(X,Y,Z):
    '''Separates the data into two subsets: labeled and non labeled.\n
       Return: A 6 tuple (X_labeled, Y_labeled, Z_labeled, X_non_labled, Y_non_lableled, Z_non_labeled)
    '''
    mask_labels = Y.sum(axis=1) >= 1

    mask_no_label = ~mask_labels

    X_labeled = X[mask_labels]
    Y_labeled = Y[mask_labels]
    Z_labeled = Z[mask_labels]

    X_non_labeled = X[mask_no_label]
    Y_non_labeled = Y[mask_no_label]
    Z_non_labeled = Z[mask_no_label]

    return X_labeled, Y_labeled, Z_labeled, X_non_labeled, Y_non_labeled, Z_non_labeled

def mono_label_to_multi(Y_mono, sample_submission=None):
    """Encode primary labels in the official multi-label target order."""
    columns = species_vocabulary(sample_submission)
    return pd.get_dummies(pd.Series(Y_mono), dtype=int).reindex(columns=columns, fill_value=0).to_numpy()


def load_data(base_path="ressource/processed_data4", raw_data_dir=None):
    """Load per-window Perch arrays, align soundscape labels and split recording groups.

    Returns nine arrays: train X/y, validation X/y, test X/y, and the unlabeled
    pseudo-label pool X/y/groups. Held-out recording groups never enter that pool.
    """
    base_path = project_path(base_path)
    raw = project_path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    targets = createLabelsWithCsv(base_path / "row_ids_train_soundscapes.npy",
                                  raw / "train_soundscapes_labels.csv",
                                  output_path=base_path / "y_labels_train_soundscapes.npy",
                                  sample_submission=raw / "sample_submission.csv")
    X_audio = np.load(base_path / "X_embeddings_perch_train_audio.npy")
    Y_audio = mono_label_to_multi(np.load(base_path / "y_labels_train_audio.npy"), raw / "sample_submission.csv")
    Z_audio = recording_groups(np.load(base_path / "row_ids_train_audio.npy"), window_ids=True)
    X_sound = np.load(base_path / "X_embeddings_perch_train_soundscapes.npy")
    Z_sound = recording_groups(np.load(base_path / "row_ids_train_soundscapes.npy"), window_ids=True)
    X_lab, Y_lab, Z_lab, X_unlab, Y_unlab, Z_unlab = partition_data(X_sound, targets, Z_sound)
    return split_grouped_data(np.concatenate((X_audio, X_lab)), np.concatenate((Y_audio, Y_lab)),
                              np.concatenate((Z_audio, Z_lab)), X_unlab, Y_unlab, Z_unlab)


def load_csv_features(features_csv="ressource/features/all_5s_features.csv", labels_csv="ressource/raw_data/train_soundscapes_labels.csv", sample_submission=None):
    """Load the upstream five-second MFCC table and split by original recording.

    The window-table schema differs from the historical whole-recording CSVs.
    Duration is retained as metadata, following the existing multi-label workflow.
    """
    frame = pd.read_csv(project_path(features_csv))
    metadata = {"row_id", "source", "filepath", "filename", "start", "end", "start_seconds", "end_seconds", "duration"}
    features = [col for col in frame if col not in metadata]
    audio, sound = frame[frame.source == "train_audio"], frame[frame.source == "train_soundscapes"]
    labels = audio.filepath.map(lambda path: str(path).replace("\\", "/").split("/")[-2]).to_numpy()
    Y_audio = mono_label_to_multi(labels, sample_submission)
    Z_audio = recording_groups(audio.filename)
    Y_sound = labels_for_windows(sound.row_id, pd.read_csv(project_path(labels_csv)), species_vocabulary(sample_submission))
    Z_sound = recording_groups(sound.filename)
    X_lab, Y_lab, Z_lab, X_unlab, Y_unlab, Z_unlab = partition_data(sound[features].to_numpy(), Y_sound, Z_sound)
    return split_grouped_data(np.concatenate((audio[features].to_numpy(), X_lab)), np.concatenate((Y_audio, Y_lab)),
                              np.concatenate((Z_audio, Z_lab)), X_unlab, Y_unlab, Z_unlab)


def _save_figure(path, **kwargs):
    """Create the requested project-relative figure directory before saving."""
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, **kwargs)


def plot_learning_curve(train_sizes, train_scores, val_scores, model_name):

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue", marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)

    plt.plot(train_sizes, val_mean, label="Validation score", color="red", marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="red", alpha=0.1)

    plt.title(f"Learning curve for {model_name}", fontsize=14)
    plt.xlabel("Training sample size")
    plt.ylabel("F1-Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    _save_figure(f"ressource/results/ml_classic/learning_curve_{model_name}.png", dpi=300)
    plt.show()

def plot_pr_curve(Y_true, Y_prob, model_name):
    precision, recall, _ = precision_recall_curve(Y_true.ravel(), Y_prob.ravel())
    avg_precision = average_precision_score(Y_true, Y_prob, average="micro")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f'Micro-average PR (AP = {avg_precision:.2f})')

    plt.title(f"Precision-Recall for {model_name}", fontsize=14)
    plt.xlabel("Recall - Birds found")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    _save_figure(f"ressource/results/ml_classic/pr_curve_{model_name}.png", dpi=300)
    plt.show()


def plot_top_flop_f1(Y_true, Y_pred, model_name):
    df_sub = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
    species_names = [str(col).strip() for col in df_sub.columns if col != 'row_id']
    f1_scores = f1_score(Y_true, Y_pred, average=None)

    score_dict = {nom: score for nom, score in zip(species_names, f1_scores)}
    sorted_scores = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)

    top_5 = sorted_scores[:5]
    flop_5 = sorted_scores[-5:]

    to_plot = flop_5 + top_5
    noms = [x[0] for x in to_plot]
    scores = [x[1] for x in to_plot]

    colors = ['#e74c3c']*5 + ['#2ecc71']*5

    plt.figure(figsize=(10, 7))
    plt.barh(noms, scores, color=colors)
    plt.axvline(x=np.mean(f1_scores), color='gray', linestyle='--', label=f'Moyenne globale ({np.mean(f1_scores):.2f})')

    plt.title(f"5 best VS 5 worst for {model_name}", fontsize=14)
    plt.xlabel("F1-Score")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    _save_figure(f"ressource/results/ml_classic/top_flop_f1_{model_name}.png", dpi=300)
    plt.show()

def plot_worst_confusion(Y_true, Y_pred, model_name):
    import seaborn as sns
    df_sub = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
    species_names = [str(col).strip() for col in df_sub.columns if col != 'row_id']
    f1_scores = f1_score(Y_true, Y_pred, average=None)
    pire_idx = np.argmin(f1_scores)
    worst_species = species_names[pire_idx]

    cm = confusion_matrix(Y_true[:, pire_idx], Y_pred[:, pire_idx])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['Missing (Prediction)', 'Present (Prediction)'],
                yticklabels=['Missing (Truth)', 'Present (Truth)'])

    plt.title(f"Error matrix: {worst_species}\n(Worst species) for {model_name}", fontsize=14)
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    _save_figure(f"ressource/results/ml_classic/confusion_worst_{model_name}.png", dpi=300)
    plt.show()

def plot_pseudo_label_tradeoff(Y_true_test, Y_proba_test, model_name):

    y_true_flat = Y_true_test.ravel()
    y_prob_flat = Y_proba_test.ravel()

    thresholds = np.linspace(0.5, 0.99, 50)
    precisions = []
    volumes = []

    for threshold in thresholds:
        pseudo_labels = (y_prob_flat >= threshold).astype(int)

        volume = np.sum(pseudo_labels)
        volumes.append(volume)

        if volume > 0:
            prec = precision_score(y_true_flat, pseudo_labels, zero_division=0)
        else:
            prec = 1.0
        precisions.append(prec)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = '#2ecc71' # Vert
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Precision', color=color, fontsize=12)
    ax1.plot(thresholds, precisions, color=color, linewidth=3, label="Precision")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.05])

    ax2 = ax1.twinx()
    color = '#e74c3c' # Rouge
    ax2.set_ylabel('Number of pseudo-labels kept', color=color, fontsize=12)
    ax2.plot(thresholds, volumes, color=color, linewidth=3, linestyle='--', label="Volume")
    ax2.tick_params(axis='y', labelcolor=color)

    chosen_threshold = 0.90
    plt.axvline(x=chosen_threshold, color='gray', linestyle=':', linewidth=2)
    plt.text(chosen_threshold + 0.01, max(volumes)*0.8, f'chosen_threshold : {chosen_threshold}', color='gray', fontsize=11, fontweight='bold')

    plt.title(f"Pseudo labeling justification for {model_name}", fontsize=14)
    fig.tight_layout()
    _save_figure(f"ressource/results/ml_classic/pseudo_label_justification_{model_name}.png", dpi=300)
    plt.show()


def plot_umap_features_justification(X_train, Y_train, top_k=5):
    import umap
    import seaborn as sns
    df_sub = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
    species_names = [col for col in df_sub.columns if col != 'row_id']

    class_counts = Y_train.sum(axis=0)
    top_classes_idx = np.argsort(class_counts)[::-1][:top_k]

    indices_to_plot = []
    labels_to_plot = []

    for class_idx in top_classes_idx:
        rows = np.where(Y_train[:, class_idx] == 1)[0]
        indices_to_plot.extend(rows)
        name = species_names[class_idx]
        labels_to_plot.extend([name] * len(rows))

    X_subset = X_train[indices_to_plot]


    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=15)
    embedding_2d = reducer.fit_transform(X_subset)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels_to_plot,
        palette="tab10",
        s=30,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.2
    )

    plt.title("Natural separability of species (UMAP on MFCC)", fontsize=15, fontweight='bold')
    plt.xlabel("Dimension UMAP 1")
    plt.ylabel("Dimension UMAP 2")

    plt.legend(title="Top species", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    _save_figure("ressource/results/umap_justification_mfcc.png", dpi=300)
    print("Graph saved : ressource/results/umap_justification_mfcc.png")
    plt.show()


def plot_umap_projection(X_true, X_pseudo, X_noise):
    import umap
    X_total = np.vstack((X_true, X_pseudo, X_noise))

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=15)
    embedding_2d = reducer.fit_transform(X_total)

    len_true = len(X_true)
    len_pseudo = len(X_pseudo)

    emb_true = embedding_2d[:len_true]
    emb_pseudo = embedding_2d[len_true : len_true + len_pseudo]
    emb_noise = embedding_2d[len_true + len_pseudo:]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    plt.scatter(emb_noise[:, 0], emb_noise[:, 1], c='#bdc3c7', s=15, alpha=0.5, label="noise")

    plt.scatter(emb_true[:, 0], emb_true[:, 1], c='#3498db', s=40, alpha=0.8, label="True label")

    plt.scatter(emb_pseudo[:, 0], emb_pseudo[:, 1], c='#e74c3c', s=80, marker='X', edgecolors='black', label="Generated pseudo-labem (>0.90)")

    plt.title("UMAP projection (Espèce Ciblée)", fontsize=15, fontweight='bold')
    plt.xlabel("Dimension UMAP 1")
    plt.ylabel("Dimension UMAP 2")
    plt.legend(loc="best", fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
    _save_figure("ressource/results/ml_classic/umap_projection.png", dpi=300)
    plt.show()


def plot_pr_comparison(Y_test, Y_proba_base, Y_proba_final, model_name):


    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 7))

    prec_base, rec_base, _ = precision_recall_curve(Y_test.ravel(), Y_proba_base.ravel())
    ap_base = average_precision_score(Y_test, Y_proba_base, average="micro")
    plt.plot(rec_base, prec_base, color="#3498db", linestyle="--", linewidth=2,
             label=f'Base model (AP = {ap_base:.3f})')

    prec_final, rec_final, _ = precision_recall_curve(Y_test.ravel(), Y_proba_final.ravel())
    ap_final = average_precision_score(Y_test, Y_proba_final, average="micro")
    plt.plot(rec_final, prec_final, color="#e74c3c", linewidth=3,
             label=f'Final model with pseudo labels (AP = {ap_final:.3f})')

    baseline_interpolated = np.interp(rec_final[::-1], rec_base[::-1], prec_base[::-1])[::-1]
    plt.fill_between(rec_final, baseline_interpolated, prec_final, where=(prec_final > baseline_interpolated),
                     color='#2ecc71', alpha=0.2, label='Performance gain')

    plt.title(f"Pseudo labeling impact using {model_name}", fontsize=15, fontweight='bold')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
    _save_figure(f"ressource/results/ml_classic/conclusion_comparison_pr_{model_name}.png", dpi=300)
    plt.show()

def plot_model_comparison(score_logreg, score_svm):

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 5))

    modeles = ['Logistic\nRegression', 'RBF\nSVM']
    scores = [score_logreg, score_svm]

    couleurs = ['#3498db', '#e67e22']

    barres = ax.bar(modeles, scores, color=couleurs, width=0.5)

    for barre in barres:
        hauteur = barre.get_height()
        ax.annotate(f'{hauteur:.3f}',
                    xy=(barre.get_x() + barre.get_width() / 2, hauteur),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title("Comparison between logreg and rbf_svm on the validation set)", fontsize=14, pad=15)
    plt.ylabel("F1-Score", fontsize=12)

    plt.ylim(0, max(scores) + 0.1)

    plt.tight_layout()
    _save_figure("ressource/results/ml_classic/comparison_rbf_log.png", dpi=300)
    plt.show()
