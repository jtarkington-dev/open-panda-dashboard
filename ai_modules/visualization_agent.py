import pandas as pd
import json
from openai import OpenAI


class VisualizationAgent:
    def __init__(self, model: OpenAI):
        self.client = model

    def run(self, df: pd.DataFrame, user_instruction: str) -> dict:
        """
        Uses GPT to generate a Plotly chart based on user instructions and a provided dataset.

        Returns:
            {
                "plotly_graph": <dict>,
                "messages": [summary]
            }
        """
        try:
            sample_data = df.head(50).to_json(orient="records")

            prompt = f"""
You are an expert Python chart generator. Return only a VALID JSON object using Plotly — no markdown, no code blocks, no comments.

USER REQUEST:
{user_instruction}

DATA SAMPLE (first 50 rows):
{sample_data}

REQUIRED OUTPUT FORMAT:
{{
  "plotly_graph": {{ Plotly chart dictionary }},
  "comment": "One sentence insight about the chart"
}}

RESTRICTIONS:
- Use plotly.express or plotly.graph_objects
- Match the requested chart type: scatter, line, pie, bar, histogram, box, etc.
- DO NOT use 'heatmapgl' or other unsupported types
- Output must be valid JSON, directly loadable with `json.loads()`
- DO NOT include explanation, markdown, or extra text — only return the JSON object
            """.strip()

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only chart generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            content = response.choices[0].message.content.strip()

            # Remove accidental code fences
            if content.startswith("```"):
                content = content.strip("```json").strip("```").strip()

            parsed = json.loads(content)

            return {
                "plotly_graph": parsed.get("plotly_graph", {}),
                "data_visualization_function": "",  # leave blank for UI control
                "messages": [parsed.get("comment", "Chart generated.")]
            }

        except Exception as e:
            return {
                "plotly_graph": {},
                "data_visualization_function": "",
                "messages": [f"⚠️ Visualization error: {str(e)}"]
            }
