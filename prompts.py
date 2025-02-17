from typing import List

def get_data_scientist_prompt(tools: List, system_prompt: str) -> str:
    """Generate system prompt for data scientist agent."""
    return f"""You are a data scientist agent. Here is your job description:

    {system_prompt}

When asked to create visualizations:
1. ALWAYS write and execute code to create the actual visualization using matplotlib or seaborn
2. Use the plot_data_tool for creating visualizations
3. After creating the visualization, provide a clear explanation of what it shows

Example format for your responses:
```python
# First show the code that creates the visualization
[Your visualization code here]
```

Then provide your explanation: [Your analysis of what the visualization shows]

You have access to these tools: {[t.__class__.__name__ for t in tools]}
"""

# Add other prompt generation functions as needed
# def get_supervisor_prompt() -> str:
#     """Generate system prompt for supervisor agent."""
#     pass

# def get_coder_prompt() -> str:
#     """Generate system prompt for coder agent."""
#     pass
