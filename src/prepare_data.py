"""Preprocesamiento centralizado del dataset NATICUSdroid.

Este módulo carga el CSV original, valida que la variable objetivo sea binaria,
crea una partición estratificada 70/30 reproducible y, si hace falta, guarda
los artefactos procesados para que todos los notebooks usen exactamente el
mismo corte de datos.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_COLUMN = "Result"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.30
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DATA_CANDIDATES = (
    ROOT_DIR / "data" / "raw" / "data.csv",
    ROOT_DIR / "data" / "data.csv",
)


@dataclass(frozen=True)
class DatasetSplitArtifacts:
    """Rutas de los artefactos generados."""

    train_features: Path
    test_features: Path
    train_target: Path
    test_target: Path
    train_full: Path
    test_full: Path
    metadata: Path


def resolve_data_path(preferred_path: Optional[str | Path] = None) -> Path:
    """Devuelve la primera ruta válida del CSV del dataset."""

    if preferred_path is not None:
        candidate = Path(preferred_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset not found at: {candidate}")

    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in RAW_DATA_CANDIDATES)
    raise FileNotFoundError(f"No dataset CSV found. Searched:\n{searched}")


def load_dataset(preferred_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Carga el dataset y normaliza detalles básicos del esquema."""

    data_path = resolve_data_path(preferred_path)
    dataframe = pd.read_csv(data_path)
    dataframe.columns = [column.strip() for column in dataframe.columns]

    if DEFAULT_TARGET_COLUMN not in dataframe.columns:
        raise KeyError(f"Target column '{DEFAULT_TARGET_COLUMN}' was not found in {data_path}")

    return dataframe


def validate_dataset(dataframe: pd.DataFrame, target_column: str = DEFAULT_TARGET_COLUMN) -> None:
    """Valida que el dataset tenga la estructura esperada para clasificación binaria."""

    missing_count = int(dataframe.isna().sum().sum())
    if missing_count != 0:
        raise ValueError(f"Dataset contains {missing_count} missing values; expected none.")

    unique_target_values = set(pd.Series(dataframe[target_column]).dropna().unique().tolist())
    if not unique_target_values.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_column}' must be binary with values 0/1. Found: {sorted(unique_target_values)}"
        )


def summarize_dataset(dataframe: pd.DataFrame, target_column: str = DEFAULT_TARGET_COLUMN) -> Dict[str, Any]:
    """Devuelve un resumen breve para notebooks e informes."""

    target_counts = dataframe[target_column].value_counts().sort_index()
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "feature_columns": int(dataframe.shape[1] - 1),
        "missing_values": int(dataframe.isna().sum().sum()),
        "class_distribution": {str(int(cls)): int(count) for cls, count in target_counts.items()},
        "class_distribution_percent": {
            str(int(cls)): round((count / len(dataframe)) * 100, 4) for cls, count in target_counts.items()
        },
    }


def split_dataset(
    dataframe: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Crea la partición estratificada y reproducible que usan todos los notebooks."""

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return X_train, X_test, y_train, y_test


def build_artifact_paths(output_dir: Path) -> DatasetSplitArtifacts:
    """Construye las rutas estándar de salida dentro de la carpeta de datos procesados."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return DatasetSplitArtifacts(
        train_features=output_dir / "X_train.csv",
        test_features=output_dir / "X_test.csv",
        train_target=output_dir / "y_train.csv",
        test_target=output_dir / "y_test.csv",
        train_full=output_dir / "train.csv",
        test_full=output_dir / "test.csv",
        metadata=output_dir / "split_metadata.json",
    )


def save_split_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Optional[str | Path] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    source_path: Optional[Path] = None,
) -> DatasetSplitArtifacts:
    """Guarda la partición train/test como CSV y también un JSON con metadatos."""

    artifacts = build_artifact_paths(Path(output_dir) if output_dir is not None else DEFAULT_PROCESSED_DIR)

    X_train.to_csv(artifacts.train_features, index=False)
    X_test.to_csv(artifacts.test_features, index=False)
    y_train.to_frame(name=target_column).to_csv(artifacts.train_target, index=False)
    y_test.to_frame(name=target_column).to_csv(artifacts.test_target, index=False)

    train_full = X_train.copy()
    train_full[target_column] = y_train.values
    test_full = X_test.copy()
    test_full[target_column] = y_test.values

    train_full.to_csv(artifacts.train_full, index=False)
    test_full.to_csv(artifacts.test_full, index=False)

    metadata = {
        "source_path": str(source_path) if source_path is not None else None,
        "target_column": target_column,
        "random_state": random_state,
        "test_size": test_size,
        "split_strategy": "stratified",
        "artifacts": {key: str(value) for key, value in asdict(artifacts).items()},
        "train_shape": [int(train_full.shape[0]), int(train_full.shape[1])],
        "test_shape": [int(test_full.shape[0]), int(test_full.shape[1])],
    }
    artifacts.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifacts


def prepare_dataset(
    preferred_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Carga, valida, divide y guarda los artefactos del dataset."""

    dataframe = load_dataset(preferred_path)
    validate_dataset(dataframe, target_column=target_column)
    X_train, X_test, y_train, y_test = split_dataset(
        dataframe,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )
    artifacts = save_split_artifacts(
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir=output_dir,
        target_column=target_column,
        random_state=random_state,
        test_size=test_size,
        source_path=resolve_data_path(preferred_path),
    )

    return {
        "summary": summarize_dataset(dataframe, target_column=target_column),
        "artifacts": asdict(artifacts),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def main() -> None:
    """Ejecuta el flujo de preprocesamiento desde la línea de comandos."""

    results = prepare_dataset()
    summary = results["summary"]
    artifacts = results["artifacts"]

    print("NATICUSdroid preprocessing complete")
    print(f"Rows: {summary['rows']:,}")
    print(f"Features: {summary['feature_columns']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Class distribution: {summary['class_distribution']}")
    print(f"Artifacts saved under: {Path(artifacts['metadata']).parent}")


if __name__ == "__main__":
    main()
