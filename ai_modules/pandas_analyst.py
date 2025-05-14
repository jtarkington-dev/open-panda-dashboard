import pandas as pd
from openai import OpenAI
from ai_modules.wrangling_agent import DataWranglingAgent
from ai_modules.visualization_agent import VisualizationAgent
from utils.text_cleanup import remove_consecutive_duplicates

class PandasAnalyst:
    def __init__(self, model, wrangler: DataWranglingAgent, visualizer: VisualizationAgent):
        self.model = model
        self.wrangler = wrangler
        self.visualizer = visualizer
        self.response = {}

    def invoke_agent(self, user_instruction: str, df: pd.DataFrame):
        self.response = {}

        try:
            intent = self._classify_intent(df.head(50), user_instruction)

            if intent == "chart":
                self.response = self.visualizer.run(df, user_instruction)
            elif intent in ["table", "insight", "summary"]:
                self.response = self.wrangler.run(df, user_instruction)
            else:
                self.response = {
                    "messages": [f"⚠️ Intent unclear. GPT classified it as: '{intent}'."]
                }

        except Exception as e:
            self.response = {"messages": [f"⚠️ Agent failure: {str(e)}"]}

    def _classify_intent(self, df_sample: pd.DataFrame, query: str) -> str:
        system_prompt = """
You are an intent classification AI for a pandas data analysis assistant.

Given a user's question and a sample of the dataset, classify the user's intent into one of the following categories:
- summary: The user wants a summary of the dataset or general overview.
- chart: The user is asking for a visualization or plot.
- table: The user is asking for tabular data or filtered records.
- insight: The user is asking for key findings, anomalies, or trends.
- unknown: The intent is unclear or unsupported.

Respond with only the category name in lowercase.
""".strip()

        try:
            response = self.model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Dataset:\n{df_sample.to_json(orient='records')}\n\nQuestion: {query}"}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            return "unknown"

    def get_plotly_graph(self):
        return self.response.get("plotly_graph")

    def get_data_wrangled(self):
        data = self.response.get("data_wrangled")
        if isinstance(data, list) and isinstance(data[0], dict):
            return pd.DataFrame(data)
        return None

    def get_data_wrangler_function(self):
        return self.response.get("data_wrangler_function", "")

    def get_data_visualization_function(self):
        return self.response.get("data_visualization_function", "")

    def get_workflow_summary(self, markdown=False):
        messages = self.response.get("messages", [])
        return "\n\n".join(messages) if markdown else messages
