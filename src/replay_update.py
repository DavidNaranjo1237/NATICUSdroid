import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from datetime import datetime


def simulate_growth(X: pd.DataFrame, y: pd.Series, n_rounds: int = 3, update_size: float = 0.05, seed: int = 42) -> Tuple[List[Tuple[pd.DataFrame, pd.Series]], pd.DataFrame, pd.Series]:
    """Simula varias rondas de llegada de nuevos datos.

    Args:
        X: características de entrenamiento.
        y: etiquetas de entrenamiento.
        n_rounds: número de rondas de actualización a simular.
        update_size: fracción del dataset original que aporta cada ronda.
        seed: semilla para reproducibilidad.

    Returns:
        updates: lista de tuplas (X_update, y_update) por cada ronda, sin reemplazo.
        X_remain, y_remain: datos que quedan después de extraer las rondas.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    X_shuffled = X.iloc[idx].reset_index(drop=True)
    y_shuffled = y.iloc[idx].reset_index(drop=True)

    updates: List[Tuple[pd.DataFrame, pd.Series]] = []
    start = 0
    n_total = len(X_shuffled)
    per_round = max(1, int(np.floor(update_size * n_total)))

    for r in range(n_rounds):
        end = min(start + per_round, n_total)
        if start >= end:
            break
        X_up = X_shuffled.iloc[start:end].reset_index(drop=True)
        y_up = y_shuffled.iloc[start:end].reset_index(drop=True)
        updates.append((X_up, y_up))
        start = end

    X_remain = X_shuffled.iloc[start:].reset_index(drop=True)
    y_remain = y_shuffled.iloc[start:].reset_index(drop=True)

    return updates, X_remain, y_remain


@dataclass
class ReplayVectorState:
    selected_features: List[str]
    replay_rate: float
    best_params: Dict
    historical_size: int
    update_size: int
    selection_method: str
    threshold: float
    min_features: int
    created_at: str


def save_vector_state(path: str, state: ReplayVectorState):
    payload = asdict(state)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(payload, f, indent=2)


def load_vector_state(path: str) -> ReplayVectorState:
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return ReplayVectorState(**data)


def build_replay_subset(X_hist: pd.DataFrame, y_hist: pd.Series, replay_rate: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Construye una muestra de replay a partir de la memoria histórica.

    Args:
        X_hist, y_hist: memoria histórica.
        replay_rate: fracción de muestras a conservar.
    """
    if replay_rate <= 0 or len(X_hist) == 0:
        return pd.DataFrame(columns=X_hist.columns), pd.Series(dtype=y_hist.dtype)

    X_replay = X_hist.sample(frac=replay_rate, random_state=seed)
    y_replay = y_hist.loc[X_replay.index]
    return X_replay.reset_index(drop=True), y_replay.reset_index(drop=True)


def select_features_advanced(X: pd.DataFrame, y: pd.Series, model: Optional[RandomForestClassifier] = None, method: str = 'combined', cumulative_threshold: float = 0.9, min_features: int = 20, random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """Selecciona variables relevantes con una regla de importancia combinada.

    Permite usar 'importance' (importancia del modelo), 'mutual_info' o 'combined' (suma ponderada).
    Devuelve una tabla con el puntaje de cada variable y la lista seleccionada.
    """
    features = list(X.columns)
    n = len(features)

    imp_scores = np.zeros(n)
    mi_scores = np.zeros(n)

    if model is not None and hasattr(model, 'feature_importances_'):
        imp_scores = np.array(model.feature_importances_)

    # La información mutua requiere valores finitos.
    try:
        mi = mutual_info_classif(X.fillna(0), y, random_state=random_state)
        mi_scores = np.array(mi)
    except Exception:
        mi_scores = np.zeros(n)

    # Normalizamos para comparar ambos puntajes en la misma escala.
    def _norm(arr):
        s = arr.sum()
        return arr / s if s > 0 else np.zeros_like(arr)

    imp_n = _norm(imp_scores)
    mi_n = _norm(mi_scores)

    if method == 'importance':
        score = imp_n
    elif method == 'mutual_info':
        score = mi_n
    else:
        score = 0.6 * imp_n + 0.4 * mi_n

    score_series = pd.Series(score, index=features).sort_values(ascending=False)
    cumulative = score_series.cumsum()
    selected = score_series.index[cumulative <= cumulative_threshold].tolist()
    if len(selected) < min_features:
        selected = score_series.head(min_features).index.tolist()

    importance_table = pd.DataFrame({
        'feature': score_series.index,
        'score': score_series.values,
        'cumulative': cumulative.values,
    })
    return importance_table, selected


def apply_warm_start_update(rf: RandomForestClassifier, X_replay: pd.DataFrame, y_replay: pd.Series, X_update: pd.DataFrame, y_update: pd.Series, add_trees: int = 50) -> RandomForestClassifier:
    """Actualiza un RandomForest con warm_start sobre replay y el lote nuevo.
    """
    X_combined = pd.concat([X_replay, X_update], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_replay, y_update], axis=0).reset_index(drop=True)

    rf.warm_start = True
    prev_n = getattr(rf, 'n_estimators', None)
    if prev_n is None:
        prev_n = 100
    rf.n_estimators = prev_n + add_trees
    rf.fit(X_combined, y_combined)
    return rf


def save_state(path: str, state: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(state, f, indent=2)


def load_state(path: str) -> Dict:
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def apply_vector(X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Filtra X con el vector de variables seleccionado y devuelve una copia."""
    missing = [c for c in selected_features if c not in X.columns]
    if missing:
        raise ValueError(f"Selected features not in X: {missing}")
    return X[selected_features].copy()


def make_working_copy_csv(original_path: str, copy_path: str):
    """Crea una copia de trabajo del CSV sin alterar el archivo original."""
    df = pd.read_csv(original_path)
    df.to_csv(copy_path, index=False)
    return copy_path
