import os
import json
import asyncio
from flask import Flask, request, jsonify, Response
import uuid
import datetime
from flask_cors import CORS
from dotenv import load_dotenv
from session_manager import SessionManager
from agents import Agent, AgentType, AgentRelation, AgentGraphManager, ANTHROPIC_API_KEY
from langchain_anthropic import ChatAnthropic

load_dotenv()

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
# Enable full error reporting
app.config['FLASK_DEBUG'] = 1

# Updated CORS configuration
CORS(app, 
     resources={r"/api/*": {
         "origins": [
             "https://team-synth-fe-git-main-dbruner23s-projects.vercel.app",
             "http://localhost:4300",
             "http://localhost:5173"
         ],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "X-Session-ID", "Accept"],
         "expose_headers": ["Content-Type", "X-Session-ID"],
         "supports_credentials": True
     }})

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
session_manager = SessionManager()

def get_session_id():
    """Get or create a session ID from the request."""
    session_id = request.headers.get('X-Session-ID')
    print(f"\n[DEBUG] Received X-Session-ID header: {session_id}")
    
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"\n[DEBUG] Generated new session ID: {session_id}")
    
    return session_id

@app.route('/api/agents', methods=['POST'])
async def create_agent():
    try:
        data = request.json
        session_id = get_session_id()
        graph_manager = session_manager.get_or_create_manager(session_id)

        # Check if API key is available with more detailed error
        if not graph_manager.llm:
            api_key = session_manager.get_api_key(session_id)
            error_msg = {
                'error': 'No API key available or LLM not initialized',
                'details': {
                    'session_id': session_id,
                    'has_api_key': bool(api_key),
                    'has_llm': bool(graph_manager.llm)
                }
            }
            print(f"\n[DEBUG] Agent creation failed: {error_msg}")
            response = jsonify(error_msg)
            response.headers['X-Session-ID'] = session_id
            return response, 400

        agent_type = AgentType(data['agent_type'])

        # Process relationships if provided, otherwise use empty list
        relationships = []
        if 'relationships' in data:
            relationships = [AgentRelation(**rel) for rel in data['relationships']]

        agent = Agent(
            id=data['id'],
            agent_type=agent_type,
            system_prompt=data['system_prompt'],
            relationships=relationships
        )

        graph_manager.add_agent(agent)

        response = jsonify({
            'id': agent.id,
            'agent_type': agent.agent_type,
            'system_prompt': data["system_prompt"],
            'relationships': [vars(rel) for rel in agent.relationships],
            'session_id': session_id
        })
        response.headers['X-Session-ID'] = session_id
        return response

    except Exception as e:
        error_response = jsonify({'error': str(e)})
        error_response.headers['X-Session-ID'] = session_id
        return error_response, 400

@app.route('/api/cancel', methods=['POST'])
def cancel_task():
    try:
        session_id = get_session_id()
        session_manager.cancel_task(session_id)
        return jsonify({
            'agent': 'system',
            'content': 'Task cancelled',
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute', methods=['POST'])
async def execute_task():
    try:
        data = request.json
        task = data.get('task')
        session_id = get_session_id()

        if not task:
            return jsonify({'error': 'No task provided'}), 400

        def generate():
            graph_manager = session_manager.get_or_create_manager(session_id)
            session_manager.start_task(session_id)

            def stream_results():
                async def async_generator():
                    try:
                        async for step in graph_manager.execute_task(task, session_id):
                            if not session_manager.is_task_active(session_id):
                                yield f"data: {json.dumps({'agent': 'system', 'content': 'Task cancelled', 'timestamp': datetime.datetime.now().isoformat()})}\n\n"
                                break
                            print(f"\n[{step['timestamp']}] Agent {step['agent']}:")
                            print(f"{step['content']}\n")
                            print("-" * 80)
                            yield f"data: {json.dumps(step)}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    finally:
                        session_manager.cancel_task(session_id)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async_gen = async_generator()
                while True:
                    try:
                        yield loop.run_until_complete(async_gen.__anext__())
                    except StopAsyncIteration:
                        break

            return stream_results()

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    session_id = get_session_id()
    graph_manager = session_manager.get_or_create_manager(session_id)

    response = jsonify({
        'agents': [
            {
                'id': agent.id,
                'type': agent.agent_type,
                'relationships': [vars(rel) for rel in agent.relationships]
            }
            for agent in graph_manager.agents.values()
        ],
        'session_id': session_id
    })
    response.headers['X-Session-ID'] = session_id
    return response

@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    try:
        data = request.json
        api_key = data.get('api_key')
        session_id = get_session_id()

        if not api_key:
            return jsonify({'error': 'No API key provided'}), 400

        print(f"\n[DEBUG] Setting API key for session {session_id}")
        # Store the API key in the session manager
        session_manager.set_api_key(session_id, api_key)
        
        # Get the manager to verify the key was set
        graph_manager = session_manager.get_or_create_manager(session_id)
        if not graph_manager.llm:
            return jsonify({'error': 'Failed to initialize LLM after setting API key'}), 500
            
        print(f"\n[DEBUG] API key set successfully for session {session_id}")
        response = jsonify({
            'message': 'API key updated successfully',
            'session_id': session_id
        })
        # Add session ID to response headers
        response.headers['X-Session-ID'] = session_id
        return response
    except Exception as e:
        print(f"\n[DEBUG] Error setting API key: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()