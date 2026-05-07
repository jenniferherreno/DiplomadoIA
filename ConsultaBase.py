import os
import json
import warnings
from typing import Dict

warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake


def load_config(path: str) -> Dict:
    """Carga el archivo config.json."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vectorstore(cfg: Dict):
    """Carga DeepLake usando la configuración del config.json."""
    dl_cfg = cfg["deeplake"]
    emb_cfg = cfg["embedding"]

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg["model_name"],
        model_kwargs={
            "device": emb_cfg.get("device", "cpu")
        },
        encode_kwargs={
            "normalize_embeddings": bool(
                emb_cfg.get("normalize_embeddings", True)
            )
        }
    )

    dataset_path = os.path.expanduser(dl_cfg["dataset_path"])

    db = DeepLake(
        dataset_path=dataset_path,
        embedding=embeddings,
        read_only=dl_cfg.get("read_only", True)
    )

    return db


def mostrar_chunks_recuperados(cfg: Dict, query: str):
    """Muestra los k chunks más similares al prompt del usuario."""
    k = int(cfg["retrieval"].get("k", 5))

    db = load_vectorstore(cfg)

    results = db.similarity_search_with_score(query, k=k)

    print("\n" + "=" * 100)
    print(f"Prompt del usuario:\n{query}")
    print("=" * 100)
    print(f"\nTop {k} chunks recuperados:\n")

    for i, (doc, score) in enumerate(results, start=1):
        print("-" * 100)
        print(f"Chunk #{i}")
        print(f"Score / similitud: {score}")
        print("-" * 100)

        print(doc.page_content)

        if doc.metadata:
            print("\nMetadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")

        print()

    print("=" * 100)


if __name__ == "__main__":
    config_file = "config.json"
    cfg = load_config(config_file)

    print("\nBuscador de chunks DeepLake")
    print("Escribe 'salir' para terminar.\n")

    while True:
        user_prompt = input("Prompt del usuario: ").strip()

        if user_prompt.lower() in {"salir", "exit", "quit"}:
            print("Fin.")
            break

        if not user_prompt:
            continue

        mostrar_chunks_recuperados(cfg, user_prompt)