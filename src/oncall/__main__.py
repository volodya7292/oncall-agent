"""Allow `python -m oncall ...` invocation.

This is what the supervisor uses to spawn the MCP child via
`sys.executable -m oncall mcp` — portable across editable, wheel, and uv-tool
installs.
"""

from .main import main


if __name__ == "__main__":
    main()
