from datasets import load_dataset
from unsloth.chat_templates import get_chat_template


DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def load():
    ds = load_dataset(DATASET)
    print(ds)


if __name__ == "__main__":
    load()
