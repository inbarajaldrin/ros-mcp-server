import json
import os
from pathlib import Path

import yaml


def _get_project_root():
    """Get the project root directory by looking for config/ros_config.json"""
    current = Path(__file__).resolve().parent.parent
    config_path = current / "config" / "ros_config.json"
    if config_path.exists():
        return current
    # Fallback: try parent directory
    parent = current.parent
    config_path = parent / "config" / "ros_config.json"
    if config_path.exists():
        return parent
    # Last resort: return current directory
    return current


def load_robot_config(robot_name: str, specs_dir: str) -> dict:
    """
    Load the robot configuration from a YAML file by robot name.

    Args:
        robot_name (str): The name of the robot.
        specs_dir (str): Directory containing robot specification files.

    Returns:
        dict: The robot configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    file_path = Path(specs_dir) / f"{robot_name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Robot config file not found: {file_path}")

    with file_path.open("r") as file:
        return yaml.safe_load(file) or {}


def parse_robot_config(name: str, specs_dir: str = "utils/robot_specifications") -> dict:
    """
    Parse the robot configuration to a more accessible format.

    Args:
        name (str): The name of the robot.
        specs_dir (str): Directory containing robot specification files.

    Returns:
        dict: Parsed robot configuration with robot name as key.
    """

    name = name.replace(" ", "_")
    config = load_robot_config(name, specs_dir)
    parsed_config = {}

    # Check if the loaded config has the required fields
    if not config:
        raise ValueError(f"No configuration found for robot '{name}'")

    # Check required fields
    for field in ("type", "prompts"):
        if field not in config or config[field] in (None, ""):
            raise ValueError(f"Robot '{name}' is missing required field: {field}")

    # Create configuration with robot name as key
    parsed_config[name] = {"type": config["type"], "prompts": config["prompts"]}

    return parsed_config


def get_robot_specifications(specs_dir: str = "utils/robot_specifications") -> dict:
    """
    Get a list of all available robot specification files.

    Args:
        specs_dir (str): Directory containing robot specification files.

    Returns:
        dict: List of available robot names that can be used with parse_robot_config.
    """
    specs_path = Path(specs_dir)

    if not specs_path.exists():
        return {"error": f"Robot specifications directory not found: {specs_path}"}

    try:
        # Find all YAML files in the specifications directory
        yaml_files = list(specs_path.glob("*.yaml"))

        if not yaml_files:
            return {"error": "No robot specification files found"}

        # Extract robot names (file names without .yaml extension)
        robot_names = [file.stem for file in yaml_files]
        robot_names.sort()  # Sort alphabetically for consistency

        return {"robot_specifications": robot_names, "count": len(robot_names)}

    except Exception as e:
        return {"error": f"Failed to read robot specifications directory: {str(e)}"}


def get_ros_config(config_path: str = None) -> dict:
    """
    Get the ROS configuration from the config file.
    
    Args:
        config_path (str): Path to the ROS config JSON file. If None, uses project root/config/ros_config.json.
    
    Returns:
        dict: The ROS configuration with defaults for missing values.
    """
    if config_path is None:
        project_root = _get_project_root()
        config_file = project_root / "config" / "ros_config.json"
    else:
        # If relative path, resolve from project root
        if not os.path.isabs(config_path):
            project_root = _get_project_root()
            config_file = project_root / config_path
        else:
            config_file = Path(config_path)
    
    if not config_file.exists():
        raise ValueError(
            f"ROS config file not found: {config_file}. "
            "Please create config/ros_config.json with required settings."
        )
    
    try:
        with config_file.open("r") as file:
            config = json.load(file)
            
            # Validate mode if present
            if "mode" in config:
                mode_lower = config["mode"].lower()
                if mode_lower not in ["real", "sim"]:
                    raise ValueError(
                        f"Invalid mode '{config['mode']}' in config file. "
                        "Must be 'real' or 'sim'."
                    )
                config["mode"] = mode_lower
            
            return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_file}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading config file {config_file}: {e}")


def get_default_mode(config_path: str = None) -> str:
    """
    Get the default mode (real/sim) from the ROS configuration file.
    
    Args:
        config_path (str): Path to the ROS config JSON file. If None, uses project root/config/ros_config.json.
    
    Returns:
        str: The default mode ('real' or 'sim') from config file.
    
    Raises:
        ValueError: If config file doesn't exist or mode is not set.
    """
    config = get_ros_config(config_path)
    mode = config.get("mode")
    if not mode:
        raise ValueError(
            "mode not found in config file. "
            "Please set mode in config/ros_config.json (must be 'real' or 'sim')."
        )
    return mode


def get_ros_setup_command(config_path: str = None) -> str:
    """
    Get the ROS setup command string for bash.
    
    Uses environment variables from ROSBridge if available, otherwise falls back to config file.
    This ensures primitive scripts use the same ROS environment as the running ROSBridge instance.
    
    Args:
        config_path (str): Path to the ROS config JSON file. If None, uses project root/config/ros_config.json.
    
    Returns:
        str: Bash command string to source ROS setup files and set domain ID.
    """
    
    # Check environment variables first (from ROSBridge connection)
    env_domain_id = os.getenv("ROS_DOMAIN_ID")
    env_ros_dist = os.getenv("ROS_DISTRO")
    
    # Get values from config file (required if not in environment)
    config = get_ros_config(config_path)
    
    # ROS distribution: use environment variable or config file (required)
    if env_ros_dist:
        ros_dist = env_ros_dist
    else:
        ros_dist = config.get("ros_distribution")
        if not ros_dist:
            raise ValueError(
                "ros_distribution not found in config file and ROS_DISTRO not set in environment. "
                "Please set ros_distribution in config/ros_config.json"
            )
    
    # ROS workspace: must be in config file
    ros_ws = config.get("ros_workspace")
    if not ros_ws:
        raise ValueError(
            "ros_workspace not found in config file. "
            "Please set ros_workspace in config/ros_config.json"
        )
    
    # Domain ID: use environment variable or config file (required)
    if env_domain_id is not None:
        domain_id = env_domain_id
    else:
        domain_id = config.get("ros_domain_id")
        if domain_id is None:
            raise ValueError(
                "ros_domain_id not found in config file and ROS_DOMAIN_ID not set in environment. "
                "Please set ros_domain_id in config/ros_config.json"
            )
    
    # Build setup command - always source ROS distro
    setup_cmd = f"source /opt/ros/{ros_dist}/setup.bash"
    
    # Source workspace (expand ~ if needed)
    if ros_ws.startswith("~"):
        ros_ws = os.path.expanduser(ros_ws)
    setup_cmd += f" && source {ros_ws}/install/setup.bash"
    
    # Only export ROS_DOMAIN_ID if it's not already set in environment
    # This ensures we use the same domain ID as ROSBridge
    if env_domain_id is None:
        setup_cmd += f" && export ROS_DOMAIN_ID={domain_id}"
    # If env_domain_id is set, we don't override it - use what ROSBridge is using
    
    return setup_cmd
