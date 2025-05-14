import plotly.io as pio

def safe_parse_plotly_json(plot_json: str):
    """
    Converts a JSON string of a Plotly figure into a Plotly object.

    Args:
        plot_json (str): A JSON string representing a Plotly figure.

    Returns:
        plotly.graph_objects.Figure or None if parsing fails.
    """
    try:
        return pio.from_json(plot_json)
    except Exception as e:
        print(f"[Plotting] Failed to parse Plotly JSON: {e}")
        return None
