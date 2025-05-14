import pandas as pd
import json
from openai import OpenAI


class DataWranglingAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, df: pd.DataFrame, user_instruction: str) -> dict:
        """
        Sends a wrangling or summary request to GPT with a sample of the data and returns
        structured results: cleaned data, Python function used, and any comments.

        Returns:
            {
                "data_wrangled": [dicts],
                "data_wrangler_function": "def transform_data(df): ...",
                "messages": [str]
            }
        """
        try:
            # Automatically format the schema summary
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                na_count = df[col].isna().sum()
                unique = df[col].nunique()
                col_info.append(f"- {col} ({dtype}), {unique} unique, {na_count} missing")

            schema_summary = "\n".join(col_info)
            sample_data = df.head(50).to_json(orient="records")

            prompt_template = f"""
You are a data wrangling and summarization AI assistant for pandas.

TASK:
Respond to the user instruction using the provided dataset. Return clean output following the JSON structure.

USER INSTRUCTION:
{user_instruction}

DATASET SCHEMA:
{schema_summary}

DATA SAMPLE (first 50 rows):
{sample_data}

RESPONSE FORMAT (strict JSON only):
{{
  "data_wrangled": [transformed dataset as list of dicts],
  "data_wrangler_function": "Python code string defining transform_data(df)",
  "comment": "Short human-readable summary of what was done"
}}

RULES:
- Use pandas only. Do NOT use markdown, code fences, or any explanations outside JSON.
- Wrap the transformation in a single function called transform_data(df).
- Do NOT write to file or plot charts.
- Your output must be valid JSON that can be parsed with json.loads().
""".strip()

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a pandas data wrangling and summarization assistant."},
                    {"role": "user", "content": prompt_template}
                ],
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.strip("```json").strip("```").strip()

            parsed = json.loads(content)

            return {
                "data_wrangled": parsed.get("data_wrangled", []),
                "data_wrangler_function": parsed.get("data_wrangler_function", ""),
                "messages": [parsed.get("comment", "")]
            }

        except Exception as e:
            return {
                "data_wrangled": [],
                "data_wrangler_function": "",
                "messages": [f"⚠️ Error during data wrangling: {str(e)}"]
            }
