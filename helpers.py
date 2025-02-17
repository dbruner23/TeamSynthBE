from typing import List
import json

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
            parsed_message['tool_calls'].append({
                'tool_name': tool_call.get('name'),
                'tool_id': tool_call.get('id'),
                'input': tool_call.get('args', {})
            })

    return parsed_message