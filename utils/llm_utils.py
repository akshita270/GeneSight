"""
LLM output utilities — JSON parsing with self-healing retry.
"""
from __future__ import annotations
import json
import re
import logging
from openai import AsyncOpenAI

logger = logging.getLogger("genesight")


async def parse_json_with_retry(
    raw: str,
    context: str,
    client: AsyncOpenAI,
    max_retries: int = 2,
) -> list | dict:
    """
    Parse LLM output as JSON.  If it fails, send the bad output back to
    GPT-4o-mini and ask it to fix the JSON — up to max_retries times.
    Raises ValueError if all attempts fail.
    """
    current = raw
    for attempt in range(max_retries + 1):
        try:
            cleaned = re.sub(r"```json|```", "", current).strip()
            # Extract outermost JSON structure if surrounded by text
            match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            if attempt >= max_retries:
                raise ValueError(
                    f"JSON parse failed after {max_retries + 1} attempts "
                    f"({context}): {exc}"
                ) from exc

            logger.warning(
                "JSON parse failed (attempt %d/%d) for %s — asking model to fix",
                attempt + 1, max_retries + 1, context,
            )
            fix_resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"The following text should be valid JSON but has an error. "
                        f"Return ONLY the corrected JSON with no explanation, "
                        f"no markdown fences, no extra text. Context: {context}\n\n"
                        f"BAD JSON:\n{current[:1500]}"
                    ),
                }],
                temperature=0,
                max_tokens=2000,
            )
            current = fix_resp.choices[0].message.content.strip()
