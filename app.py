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
CORS(app)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
session_manager = SessionManager()

def get_session_id():
    """Get or create a session ID from the request."""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

@app.route('/api/agents', methods=['POST'])
async def create_agent():
    try:
        data = request.json
        session_id = get_session_id()
        graph_manager = session_manager.get_or_create_manager(session_id)

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

        return jsonify({
            'id': agent.id,
            'agent_type': agent.agent_type,
            'system_prompt': agent.system_prompt,
            'relationships': [vars(rel) for rel in agent.relationships],
            'session_id': session_id  # Return the session ID to the client
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/cancel', methods=['POST'])
def cancel_task():
    try:
        session_id = get_session_id()
        session_manager.cancel_task(session_id)
        return jsonify({'message': 'Task cancellation requested'})
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

    return jsonify({
        'agents': [
            {
                'id': agent.id,
                'type': agent.agent_type,
                'relationships': [vars(rel) for rel in agent.relationships]
            }
            for agent in graph_manager.agents.values()
        ],
        'session_id': session_id  # Return the session ID to the client
    })

@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    try:
        data = request.json
        api_key = data.get('api_key')

        if not api_key:
            return jsonify({'error': 'No API key provided'}), 400

        # Update the API key in the agents module
        global ANTHROPIC_API_KEY
        ANTHROPIC_API_KEY = api_key

        return jsonify({'message': 'API key updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)