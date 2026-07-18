import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tomllib

import requests

__all__ = ["get_chat_completion"]

# Load settings file
settings_path = Path("settings.toml")
with settings_path.open("rb") as settings_file:
    SETTINGS = tomllib.load(settings_file)

#load API key
def load_api_key(key_file):
    """Load API key from a local file."""
    with open(key_file, "r", encoding="utf-8") as f:
        return f.read().strip()
api_key = load_api_key("api_key.txt")

def parse_args() -> argparse.Namespace:
    """Parse command-line input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=Path, help="Path to the input file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the formatted JSON result",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    file_content = args.file_path.read_text("utf-8")
    result = json.dumps(get_chat_completion(file_content), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{result}\n", encoding="utf-8")
    print(result)


def get_chat_completion(content: str) -> dict:
    """Classify each regex-delimited conversation with self-consistency."""
    sample_count = SETTINGS["general"]["sample_count"]
    blocks = _split_conversation_blocks(content)
    conversations = _parse_input_conversations(content)
    result = {"negative": [], "positive": []}
    for block, conversation in zip(blocks, conversations, strict=True):
        label = _classify_conversation(block, sample_count)
        result[label].append(conversation)
    return result


def _classify_conversation(block: str, sample_count: int) -> str:
    """Collect valid stochastic labels for one conversation and vote."""
    labels = []
    max_attempts = sample_count * 10
    for _ in range(max_attempts):
        try:
            labels.append(_request_completion(block, 1)[0])
        except ValueError:
            continue
        if len(labels) == sample_count:
            break
    if len(labels) != sample_count:
        raise RuntimeError(
            f"Could not obtain {sample_count} valid JSON samples for one conversation."
        )
    return Counter(labels).most_common(1)[0][0]


def _request_completion(content: str, conversation_count: int) -> list[str]:
    """Send one stochastic request and parse its JSON response."""
    url = "https://aigc-api.hkust-gz.edu.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }
    data = {
        "model": SETTINGS["general"]["model"],
        "messages": _assemble_chat_messages(content, conversation_count),
        "temperature": SETTINGS["general"]["temperature"],
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    response_json = response.json()
    if "choices" not in response_json:
        raise RuntimeError(f"The API did not return a completion: {response_json}")
    message = response_json["choices"][0]["message"]["content"]
    return _parse_classification(message, conversation_count)


def _parse_classification(message: str, conversation_count: int) -> list[str]:
    """Validate a model response containing one label for each conversation."""
    message = message.strip()
    if message.startswith("```"):
        message = re.sub(r"^```(?:json)?\s*|\s*```$", "", message).strip()
    try:
        classification = json.loads(message)
    except json.JSONDecodeError as error:
        raise ValueError(f"The model did not return valid JSON: {message!r}") from error

    sentiments = classification.get("sentiments") if set(classification) == {"sentiments"} else None
    if not isinstance(sentiments, list) or len(sentiments) != conversation_count:
        raise ValueError(f"Unexpected JSON result: {classification!r}")
    if any(sentiment not in {"negative", "positive"} for sentiment in sentiments):
        raise ValueError("Each sentiment must be negative or positive.")
    return sentiments


def _parse_input_conversations(content: str) -> list[dict]:
    """Convert sanitized chats to the JSON conversation shape expected by the task."""
    conversations = []
    for block in _split_conversation_blocks(content):
        lines = []
        date = None
        for raw_line in block.splitlines():
            match = re.match(r"^\[(?P<speaker>[^]]+)\]\s+(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<text>.*)$", raw_line)
            if match is None:
                raise ValueError(f"Invalid chat line: {raw_line}")
            date = date or match["date"]
            prefix = "A" if match["speaker"].lower().startswith(("agent", "support")) else "C"
            lines.append(f"{prefix}: {match['text']}")
        conversations.append({"date": date, "conversation": lines})
    return conversations


def _split_conversation_blocks(content: str) -> list[str]:
    """Split chat conversations on one or more blank lines using a regex."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content.strip()) if block.strip()]
    if not blocks:
        raise ValueError("The input does not contain any conversations.")
    return blocks


def _assemble_chat_messages(content: str, conversation_count: int) -> list[dict]:
    """Combine prompt examples and one conversation into chat messages."""
    messages = [
        {"role": "system", "content": SETTINGS["prompts"]["role_prompt"]},
        {"role": "user", "content": SETTINGS["prompts"]["negative_example"]},
        {
            "role": "system",
            "content": SETTINGS["prompts"]["negative_reasoning"],
        },
        {
            "role": "assistant",
            "content": SETTINGS["prompts"]["negative_output"],
        },
        {"role": "user", "content": SETTINGS["prompts"]["positive_example"]},
        {
            "role": "system",
            "content": SETTINGS["prompts"]["positive_reasoning"],
        },
        {
            "role": "assistant",
            "content": SETTINGS["prompts"]["positive_output"],
        },
        {"role": "user", "content": f">>>>>\n{content}\n<<<<<"},
        {
            "role": "user",
            "content": (
                f"{SETTINGS['prompts']['instruction_prompt']}\n"
                "The delimited content contains exactly one conversation, so the "
                "sentiments array must contain exactly one label."
            ),
        },
    ]
    return messages


if __name__ == "__main__":
    main(parse_args())
