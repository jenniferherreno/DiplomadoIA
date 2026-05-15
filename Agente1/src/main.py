import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

PROJECT_ROOT = Path("/Users/orodriguez/PycharmProjects/DiplomadoIA/Agente1")
sys.path.append(str(PROJECT_ROOT))

from services.load_config import load_config
from services.load_json import load_json
from services.load_text_file import load_text_file
from src.brain import run_agent_loop


def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.json"))

    load_dotenv(config["paths"]["env_file"])

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")

    llm_config = config["llm"]

    llm = ChatOpenAI(
        openai_api_base=llm_config["api_base"],
        openai_api_key=api_key,
        model_name=llm_config["model_name"],
        temperature=float(llm_config["temperature"])
    )

    memory = ConversationBufferMemory(return_messages=True)

    classifier_prompt = load_text_file(
        config["paths"]["classifier_prompt"]
    )

    options = load_json(
        config["paths"]["decision_options"]
    )

    personality = load_text_file(
        config["paths"]["personality"]
    )

    run_agent_loop(
        llm=llm,
        memory=memory,
        config=config,
        classifier_prompt=classifier_prompt,
        options=options,
        personality=personality
    )


if __name__ == "__main__":
    main()