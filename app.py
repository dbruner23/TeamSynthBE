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
             "https://team-synth-fe-eta.vercel.app",
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
        session_manager.reset_task_state(session_id)
        
        print(f"\n[DEBUG] Task cancelled and reset for session {session_id}")
        
        return jsonify({
            'agent': 'system',
            'content': 'Task cancelled',
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'cancelled',
            'ready_for_new_task': True
        })
    except Exception as e:
        print(f"\n[ERROR] Error cancelling task: {str(e)}")
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
            try:
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
                            error_message = str(e)
                            print(f"\n[ERROR] Task execution error: {error_message}")
                            error_data = {
                                'agent': 'system', 
                                'content': f'Error: {error_message}',
                                'timestamp': datetime.datetime.now().isoformat(),
                                'error': True
                            }
                            yield f"data: {json.dumps(error_data)}\n\n"
                        finally:
                            # Always ensure task is marked as complete/cancelled
                            session_manager.cancel_task(session_id)
                            session_manager.reset_task_state(session_id)

                    loop = None
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        async_gen = async_generator()
                        
                        while True:
                            try:
                                yield loop.run_until_complete(async_gen.__anext__())
                            except StopAsyncIteration:
                                break
                            except MemoryError:
                                # Explicitly handle memory errors
                                memory_error_data = {
                                    'agent': 'system',
                                    'content': 'Task failed due to insufficient memory. You can cancel this task and try again with a simpler request.',
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'error': True,
                                    'memory_error': True
                                }
                                yield f"data: {json.dumps(memory_error_data)}\n\n"
                                # Clean up resources and cancel the task
                                session_manager.cancel_task(session_id)
                                session_manager.reset_task_state(session_id)
                                break
                            except Exception as e:
                                # Handle other unexpected errors during execution
                                error_type = type(e).__name__
                                error_msg = str(e)
                                print(f"\n[ERROR] Error in execute_task: {error_type}: {error_msg}")
                                
                                # Check if it's a memory-related error (could be system exit from OOM killer)
                                if "SystemExit" in error_type or "MemoryError" in error_type or "out of memory" in error_msg.lower():
                                    memory_error_data = {
                                        'agent': 'system',
                                        'content': 'Task failed due to insufficient memory. You can cancel this task and try again with a simpler request.',
                                        'timestamp': datetime.datetime.now().isoformat(),
                                        'error': True,
                                        'memory_error': True
                                    }
                                    yield f"data: {json.dumps(memory_error_data)}\n\n"
                                else:
                                    general_error_data = {
                                        'agent': 'system',
                                        'content': f'Unexpected error: {error_type}: {error_msg}',
                                        'timestamp': datetime.datetime.now().isoformat(),
                                        'error': True
                                    }
                                    yield f"data: {json.dumps(general_error_data)}\n\n"
                                
                                # Always ensure task is cancelled on error
                                session_manager.cancel_task(session_id)
                                session_manager.reset_task_state(session_id)
                                break
                    finally:
                        # Clean up the event loop resources
                        if loop:
                            try:
                                loop.close()
                            except Exception as e:
                                print(f"\n[ERROR] Error closing event loop: {str(e)}")

                return stream_results()
            except Exception as e:
                # Handle initialization errors
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"\n[ERROR] Error initializing task: {error_type}: {error_msg}")
                
                # Always ensure task is cancelled on error
                session_manager.cancel_task(session_id)
                session_manager.reset_task_state(session_id)
                
                return f"data: {json.dumps({'agent': 'system', 'content': f'Error initializing task: {error_msg}', 'error': True, 'timestamp': datetime.datetime.now().isoformat()})}\n\n"

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
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n[ERROR] Exception in execute_task route: {error_type}: {error_msg}")
        
        # Ensure task is cancelled if an exception occurs
        if 'session_id' in locals():
            session_manager.cancel_task(session_id)
            session_manager.reset_task_state(session_id)
            
        return jsonify({
            'error': str(e),
            'error_type': error_type,
            'recoverable': True  # Allow frontend to retry
        }), 500

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

@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    """Reset a session that has experienced errors, particularly memory errors.
    
    This endpoint allows the frontend to force a complete recreation of the session's 
    graph manager, which can help recover from severe errors including memory issues.
    """
    try:
        session_id = get_session_id()
        
        # Cancel any active task first
        session_manager.cancel_task(session_id)
        
        # Use the more aggressive recovery method
        session_manager.force_recreate_manager(session_id)
        
        print(f"\n[DEBUG] Session {session_id} has been fully reset")
        
        return jsonify({
            'status': 'success',
            'message': 'Session has been reset successfully. You may now start a new task.',
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n[ERROR] Error resetting session: {error_type}: {error_msg}")
        return jsonify({
            'status': 'error',
            'error': error_msg,
            'error_type': error_type,
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

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