"""Generate a deterministic synthetic multilingual multilabel dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def generate(n_documents: int = 120) -> pd.DataFrame:
    labels = ["Ciencias Informáticas", "Educación", "Historia", "Química"]
    rows = []
    for index in range(n_documents):
        first = labels[index % len(labels)]
        second = labels[(index + 1) % len(labels)] if index % 5 == 0 else None
        spanish = index % 3 != 0
        if spanish:
            abstract = f"Esta investigación estudia métodos de {first} para la educación y la sociedad."
            fulltext = f"El trabajo presenta resultados reproducibles sobre {first}. " * 8
        else:
            abstract = f"The research study presents methods for {first} and education."
            fulltext = f"The paper reports reproducible results about {first}. " * 8
        targets = first if second is None else f"{first}||{second}"
        rows.append(
            {
                "handle": f"10915/{100000 + index}",
                "dc.description.abstract[es]" if spanish else "dc.description.abstract[en]": abstract,
                "dc.subject[es]" if spanish else "dc.subject[en]": f"{first}::http://example.org/{index}||repositorios",
                "sedici.subject.materias[es]": targets,
                "fulltext": fulltext,
            }
        )
    return pd.DataFrame(rows).fillna("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample/dummy_metadata.csv")
    parser.add_argument("--documents", type=int, default=120)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(args.documents).to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()

