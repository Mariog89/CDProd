"""Valida la estructura de carpetas y archivos obligatorios del repositorio.

Usado como etapa del pipeline de Jenkins (Jenkinsfile): falla con código 1
si falta cualquier archivo o carpeta del contrato de Stage 3, o si se
versionaron datos/artefactos, de modo que un merge que rompa la estructura no
avance a pruebas ni a aprobación.
"""

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

DIRECTORIOS_REQUERIDOS = (
    "mlops_pipeline",
    "mlops_pipeline/src",
    "scripts",
    "tests",
    ".github/workflows",
)

ARCHIVOS_REQUERIDOS = (
    "mlops_pipeline/src/Cargar_datos.py",
    "mlops_pipeline/src/comprension_eda.ipynb",
    "mlops_pipeline/src/ft_engineering.py",
    "mlops_pipeline/src/hueristic_model.py",
    "mlops_pipeline/src/model_training.py",
    "mlops_pipeline/src/model_evaluation.py",
    "mlops_pipeline/src/model_deploy.py",
    "mlops_pipeline/src/model_monitoring.py",
    "mlops_pipeline/src/config.json",
    "scripts/export_model_artifact.py",
    "scripts/validate_metrics.py",
    "app.py",
    "Dockerfile",
    "Jenkinsfile",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "readme.md",
    "set_up.bat",
)

# Patrones que nunca deben versionarse: datos locales, modelos y salidas.
PATRONES_NO_VERSIONABLES = (
    "*.csv",
    "*.joblib",
    "artifacts/*",
    "runtime/*",
    "codigoproyecto-venv/*",
)


def archivos_versionados(raiz: Path) -> list[str] | None:
    """Lista los archivos rastreados por Git, o None si no es un repo Git."""
    try:
        salida = subprocess.run(
            ["git", "ls-files"],
            cwd=raiz,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [linea for linea in salida.stdout.splitlines() if linea]


def validar_estructura(raiz: Path, versionados: list[str] | None = None) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacía si todo está bien)."""
    problemas = []
    for rel in DIRECTORIOS_REQUERIDOS:
        if not (raiz / rel).is_dir():
            problemas.append(f"falta directorio: {rel}")
    for rel in ARCHIVOS_REQUERIDOS:
        if not (raiz / rel).is_file():
            problemas.append(f"falta archivo: {rel}")
    tests = sorted((raiz / "tests").glob("test_*.py")) if (raiz / "tests").is_dir() else []
    if not tests:
        problemas.append("tests/ no contiene archivos test_*.py")
    for ruta in versionados or []:
        if any(fnmatch.fnmatch(ruta, patron) for patron in PATRONES_NO_VERSIONABLES):
            problemas.append(f"archivo versionado que no debería estar en Git: {ruta}")
    return problemas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Valida la estructura del repositorio CDProd.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Raíz del repositorio (default: carpeta padre de scripts/).",
    )
    args = parser.parse_args(argv)
    raiz = Path(args.root)
    problemas = validar_estructura(raiz, archivos_versionados(raiz))
    total = len(DIRECTORIOS_REQUERIDOS) + len(ARCHIVOS_REQUERIDOS)
    if problemas:
        print(f"Estructura inválida ({len(problemas)} problemas):")
        for problema in problemas:
            print(f"  - {problema}")
        return 1
    print(f"Estructura OK: {total} rutas obligatorias verificadas en {raiz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
