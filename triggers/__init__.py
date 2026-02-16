"""Trigger tools for MCP server.

Triggers are lightweight signal tools that return structured data to the client.
Unlike primitives and queries, triggers do not use subprocess calls - they are
importable Python modules with async handler functions.
"""
