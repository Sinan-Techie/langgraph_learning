def build_llm_prompt_batch(batch_data: list[dict]) -> str:
    """
    batch_data = [
      {
        "query": "...",
        "candidates": [
          {
            "product_id": "...",
            "product_name": "...",
            "category": "...",
            "distance": 0.123,
            "description": "..."
          }
        ]
      }
    ]
    """

    blocks = []

    for idx, item in enumerate(batch_data, start=1):
        candidate_block = "\n".join(
            [
                f"""
Candidate {i+1}:
- Product ID: {c['product_id']}
- Product Name: {c['product_name']}
- Category: {c['category']}
- Distance Score: {c['distance']}
- Description: {c['description']}
""".strip()
                for i, c in enumerate(item["candidates"])
            ]
        )

        blocks.append(
            f"""
Query {idx}:
User Query:
"{item['query']}"

Retrieved Candidates:
{candidate_block}
""".strip()
        )

    joined_blocks = "\n\n".join(blocks)

    return f"""
You are a product catalog matching assistant.

Task:
- For EACH query below, select the SINGLE best matching product
- Use ONLY the retrieved candidates
- Do NOT invent products
- If no candidate is suitable, return nulls

{joined_blocks}

Return STRICT JSON ONLY in the following format:

[
  {{
    "input_query": "<query text>",
    "selected_product_id": "<string or null>",
    "selected_product_name": "<string or null>",
    "confidence": "high | medium | low",
    "reason": "<short explanation>"
  }}
]

Rules:
- Output MUST be valid JSON
- Output MUST be a JSON array
- No markdown
- No extra text
""".strip()


def build_input_normalization_prompt(text: str) -> str:
    return f"""
You are given text that may contain multiple sentences or paragraphs listing items.
The items may be separated by commas, the word "and", or mixed punctuation, and may
appear anywhere in the paragraph.

Input variable:
{text}

Your task is to:
- Extract all distinct item names from the text, including product names with serial numbers and model names
- Remove connector words like "and", "or", etc.
- Ignore sentence structure and extra words
- Fix obvious spelling mistakes
- Preserve complete product identifiers (model numbers, serial numbers, product codes)

Output rules:
- Output ONLY the cleaned item names
- One item per line
- No numbering
- No bullets
- No explanations
- Keep model numbers and serial numbers attached to their product names
- Do NOT add extra words
- Do NOT merge unrelated items
- Do NOT output JSON

Example:

Input:
We need Intel Core i7-13700K processor, ASUS ROG Strix Z790 motherboard and Corsair Vengeance DDR5 32GB RAM for the build. Also add a Samsung 980 PRO 1TB SSD and NVIDIA GeForce RTX 4080, along with Cooler Master MasterLiquid 240mm AIO cooler and Seasonic Focus GX-850 power suply for testing.

Expected output:
Intel Core i7-13700K processor
ASUS ROG Strix Z790 motherboard
Corsair Vengeance DDR5 32GB RAM
Samsung 980 PRO 1TB SSD
NVIDIA GeForce RTX 4080
Cooler Master MasterLiquid 240mm AIO cooler
Seasonic Focus GX-850 power supply
""".strip()
