#!/usr/bin/env python3
"""
Auto-generate primitives_api.py from MCP server tool definitions.

This script discovers all tools from server.py and generates the primitives API
automatically, eliminating manual duplication.
"""

import os
import sys
import inspect
import ast
from typing import Dict, List, Any
from pathlib import Path

# Add parent directory to path (go up one level from utils/)
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

def extract_function_body(func) -> List[str]:
    """Extract the body of a function (without decorators, signature, and docstring)."""
    source_lines = inspect.getsource(func).split('\n')
    body_lines = []
    in_docstring = False
    docstring_quotes = None
    found_body_start = False
    past_def_line = False

    for i, line in enumerate(source_lines):
        stripped = line.strip()

        # Skip decorators and function definition
        if not past_def_line:
            if stripped.startswith('@'):
                continue
            if stripped.startswith('def '):
                past_def_line = True
                continue

        if not past_def_line:
            continue

        # Detect and skip docstring
        if not found_body_start:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring_quotes = '"""' if stripped.startswith('"""') else "'''"
                    # Check if docstring ends on same line
                    if stripped.count(docstring_quotes) >= 2:
                        in_docstring = False
                        found_body_start = True
                    continue
                elif stripped.endswith(docstring_quotes):
                    in_docstring = False
                    found_body_start = True
                    continue
            elif in_docstring:
                continue
            elif stripped and not stripped.startswith('#'):
                # Found first real code line
                found_body_start = True

        if found_body_start:
            body_lines.append(line)

    return body_lines

def extract_tool_implementation(source_code: str, tool_name: str) -> Dict[str, Any]:
    """Extract implementation details for a specific tool from source code."""
    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == tool_name:
            # Get the function body
            # Look for _run_primitive or _run_query calls (might be in return or assignment)
            for stmt in node.body:
                # Pattern 1: return _run_primitive(...)
                if isinstance(stmt, ast.Return):
                    if isinstance(stmt.value, ast.Call):
                        call = stmt.value
                        if isinstance(call.func, ast.Name):
                            func_name = call.func.id

                            # Extract arguments
                            args = []
                            for arg in call.args:
                                if isinstance(arg, ast.Constant):
                                    args.append(repr(arg.value))
                                elif isinstance(arg, ast.JoinedStr):
                                    # f-string
                                    args.append(ast.unparse(arg))
                                else:
                                    args.append(ast.unparse(arg))

                            # Extract keyword arguments
                            kwargs = {}
                            for keyword in call.keywords:
                                kwargs[keyword.arg] = ast.unparse(keyword.value)

                            return {
                                'helper_func': func_name,
                                'args': args,
                                'kwargs': kwargs,
                                'is_verify': False
                            }

                # Pattern 2: primitive_result = _run_primitive(...) (for verify functions)
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.value, ast.Call):
                        call = stmt.value
                        if isinstance(call.func, ast.Name) and call.func.id == '_run_primitive':
                            # This is a verify function pattern
                            args = []
                            for arg in call.args:
                                if isinstance(arg, ast.Constant):
                                    args.append(repr(arg.value))
                                elif isinstance(arg, ast.JoinedStr):
                                    args.append(ast.unparse(arg))
                                else:
                                    args.append(ast.unparse(arg))

                            kwargs = {}
                            for keyword in call.keywords:
                                kwargs[keyword.arg] = ast.unparse(keyword.value)

                            return {
                                'helper_func': '_run_primitive',
                                'args': args,
                                'kwargs': kwargs,
                                'is_verify': True
                            }

    return None

def get_function_signature(func) -> str:
    """Get the function signature as a string."""
    sig = inspect.signature(func)
    params = []

    for name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            # Required parameter
            if param.annotation != inspect.Parameter.empty:
                params.append(f"{name}: {param.annotation.__name__}")
            else:
                params.append(name)
        else:
            # Optional parameter with default
            if param.annotation != inspect.Parameter.empty:
                if isinstance(param.default, str):
                    params.append(f'{name}: {param.annotation.__name__} = "{param.default}"')
                else:
                    params.append(f"{name}: {param.annotation.__name__} = {param.default}")
            else:
                if isinstance(param.default, str):
                    params.append(f'{name} = "{param.default}"')
                else:
                    params.append(f"{name} = {param.default}")

    return f"def {func.__name__}({', '.join(params)})"

def generate_api_from_server():
    """Generate primitives_api.py from server.py tool definitions."""

    # Import server to get access to MCP tools
    import server

    # Read server.py source code (go up one level from utils/)
    server_path = Path(__file__).parent.parent / "server.py"
    with open(server_path, 'r') as f:
        server_source = f.read()

    # Get all tools from the MCP server
    # The mcp object has a registry of tools
    tools = {}

    # Find all functions decorated with @mcp.tool()
    for name, obj in inspect.getmembers(server):
        if inspect.isfunction(obj) and hasattr(obj, '__wrapped__'):
            # This is likely an MCP tool
            # Skip execute_policy_code and execute_python_code
            if name in ['execute_policy_code', 'execute_python_code']:
                continue

            tools[name] = {
                'function': obj,
                'signature': get_function_signature(obj),
                'docstring': inspect.getdoc(obj) or "",
                'implementation': extract_tool_implementation(server_source, name)
            }

    # Also manually check for known tools by looking at server module
    known_tools = [
        'get_topics', 'capture_camera_image', 'read_topic',
        'get_scene_info',
        'move_home', 'control_gripper', 'scan_workspace',
        'move_to_grasp', 'move_to_regrasp', 'translate_object', 'rotate_object',
        'verify_grasp', 'verify_assembly', 'verify_disassembly'
    ]

    for tool_name in known_tools:
        if hasattr(server, tool_name) and tool_name not in tools:
            func = getattr(server, tool_name)
            if callable(func):
                tools[tool_name] = {
                    'function': func,
                    'signature': get_function_signature(func),
                    'docstring': inspect.getdoc(func) or "",
                    'implementation': extract_tool_implementation(server_source, tool_name)
                }

    # Generate the API file
    output = []
    output.append('"""')
    output.append('Auto-generated Robot Primitives API')
    output.append('')
    output.append('This file is AUTO-GENERATED by generate_tools_api.py')
    output.append('DO NOT EDIT MANUALLY - changes will be overwritten')
    output.append('')
    output.append('To regenerate: python generate_tools_api.py')
    output.append('"""')
    output.append('')
    output.append('import subprocess')
    output.append('import os')
    output.append('import sys')
    output.append('import threading')
    output.append('import time')
    output.append('from typing import Dict, Any, Optional, List')
    output.append('from datetime import datetime')
    output.append('')
    output.append('# Get the project root directory (two levels up from primitives/utils/)')
    output.append('SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
    output.append('')

    # Include helper functions (copy from server.py)
    output.append('# Helper functions (copied from server.py)')
    output.append('')

    # Extract helper functions from server
    helper_funcs = ['_run_primitive', '_run_query', '_parse_verify_result']
    for helper_name in helper_funcs:
        if hasattr(server, helper_name):
            helper_source = inspect.getsource(getattr(server, helper_name))
            # Fix path references: replace script_dir with SCRIPT_DIR for correct paths
            # In server.py, script_dir is project root; in primitives_api.py, SCRIPT_DIR is project root
            helper_source = helper_source.replace('script_dir = os.path.dirname(os.path.abspath(__file__))', '# Using global SCRIPT_DIR instead of local script_dir')
            helper_source = helper_source.replace('script_dir', 'SCRIPT_DIR')
            # Remove redundant imports since they're at module level
            helper_source = helper_source.replace('    import subprocess\n', '')
            helper_source = helper_source.replace('    import os\n', '')
            output.append(helper_source)
            output.append('')

    # Categorize tools
    query_tools = [n for n in tools.keys() if n.startswith('get_')]
    primitive_tools = [n for n in tools.keys() if n not in query_tools and not n.startswith('verify_')]
    verify_tools = [n for n in tools.keys() if n.startswith('verify_')]
    other_tools = [n for n in tools.keys() if n not in query_tools and n not in primitive_tools and n not in verify_tools]

    # Generate query functions
    if query_tools:
        for tool_name in sorted(query_tools):
            tool = tools[tool_name]
            output.append(f"{tool['signature']} -> Dict[str, Any]:")
            # Copy the full function body
            body_lines = extract_function_body(tool['function'])
            for line in body_lines:
                output.append(line)
            output.append('')

    # Generate primitive functions
    if primitive_tools:
        for tool_name in sorted(primitive_tools):
            tool = tools[tool_name]
            output.append(f"{tool['signature']} -> Dict[str, Any]:")
            # Copy the full function body
            body_lines = extract_function_body(tool['function'])
            for line in body_lines:
                output.append(line)
            output.append('')

    # Generate verification functions
    if verify_tools:
        for tool_name in sorted(verify_tools):
            tool = tools[tool_name]
            output.append(f"{tool['signature']} -> Dict[str, Any]:")
            # Copy the full function body
            body_lines = extract_function_body(tool['function'])
            for line in body_lines:
                output.append(line)
            output.append('')

    # Write the generated file
    output_path = script_dir / "primitives" / "utils" / "primitives_api.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(output))

    print(f"✓ Generated primitives_api.py at {output_path}")
    print(f"✓ Found {len(tools)} tools")
    print(f"  - Query functions: {len(query_tools)}")
    print(f"  - Primitive functions: {len(primitive_tools)}")
    print(f"  - Verification functions: {len(verify_tools)}")

    return output_path

if __name__ == "__main__":
    try:
        output_path = generate_api_from_server()
        print("\nAPI generation complete!")
        print(f"\nNext steps:")
        print(f"1. Review the generated file: {output_path}")
        print(f"2. Fix any 'NotImplementedError' entries manually")
        print(f"3. Test with: python -c 'from primitives.utils.primitives_api import *'")
    except Exception as e:
        print(f"Error generating API: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
