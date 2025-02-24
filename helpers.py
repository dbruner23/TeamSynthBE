from typing import List
import json
import io
import base64
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import matplotlib.pyplot as plt


def parse_ai_message(message):
    """
    Parse an AIMessage object and extract its components into a standardized structure.

    Args:
        message: An AIMessage object containing content and metadata

    Returns:
        dict: A standardized structure containing the parsed message components
    """
    parsed_message = {
        'id': message.id,
        'type': 'ai_message',
        'content': [],
        'code_blocks': [],
        'tool_calls': [],
        'plots': [],  # New field for plot data
        'metadata': {
            'model': message.response_metadata.get('model'),
            'stop_reason': message.response_metadata.get('stop_reason'),
            'usage': message.response_metadata.get('usage', {})
        }
    }

    # Handle different content types
    if isinstance(message.content, str):
        # If content is a string, treat it as text content
        parsed_message['content'].append({
            'type': 'text',
            'text': message.content
        })
    elif isinstance(message.content, list):
        # Handle list of content items
        for item in message.content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    parsed_message['content'].append({
                        'type': 'text',
                        'text': item.get('text', '')
                    })
                elif item.get('type') == 'tool_use':
                    # Extract code from tool use
                    code = item.get('input', {}).get('code', '')
                    if code:
                        parsed_message['code_blocks'].append({
                            'language': 'python',  # Assuming Python for now
                            'code': code
                        })
                    parsed_message['tool_calls'].append({
                        'tool_name': item.get('name'),
                        'tool_id': item.get('id'),
                        'input': item.get('input', {})
                    })

    # Extract code blocks from tool_calls if present
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            if 'code' in tool_call.get('args', {}):
                parsed_message['code_blocks'].append({
                    'language': 'python',  # Assuming Python for now
                    'code': tool_call['args']['code']
                })

            # Handle plot data tool calls
            if tool_call.get('tool_name') == 'plot_data_tool' and 'input' in tool_call.get('args', {}):
                try:
                    # Create a new figure for this plot
                    plt.figure()

                    # Get the plot code
                    plot_code = tool_call['args']['input'].get('code', '')

                    # Execute the plot code in a local namespace to avoid conflicts
                    local_ns = {}
                    exec(plot_code, globals(), local_ns)

                    # Save the plot to a bytes buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight')
                    buffer.seek(0)

                    # Convert to base64
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    # Add plot data to the response
                    parsed_message['plots'].append({
                        'image_data': image_base64,
                        'format': 'png',
                        'tool_id': tool_call.get('id'),
                    })

                    # Clean up
                    plt.close()
                    buffer.close()
                except Exception as e:
                    print(f"Error generating plot: {str(e)}")
                    # Add error information to the plots array
                    parsed_message['plots'].append({
                        'error': str(e),
                        'tool_id': tool_call.get('id')
                    })

            parsed_message['tool_calls'].append({
                'tool_name': tool_call.get('name'),
                'tool_id': tool_call.get('id'),
                'input': tool_call.get('args', {})
            })

    return parsed_message