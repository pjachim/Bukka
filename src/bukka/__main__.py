"""Bukka CLI - Machine Learning Project Scaffolding Tool.

This module provides the command-line interface for creating and configuring
Bukka ML projects with automatic pipeline generation.
"""
import argparse
import sys
from pathlib import Path
from bukka.project import Project
from bukka.utils import bukka_logger
from bukka.cli_config import (
    BukkaConfig,
    ConfigManager,
    ConfigValidator,
    SUPPORTED_BACKENDS,
    PROBLEM_TYPES,
)

logger = bukka_logger.BukkaLogger(__name__)


def print_project_success(config: BukkaConfig) -> None:
    """Print formatted success message after project creation.
    
    Args:
        config: The project configuration used.
    
    Examples:
        >>> config = BukkaConfig(name="test", dataset="data.csv")
        >>> print_project_success(config)  # doctest: +SKIP
    """
    print(f"\n{'='*60}")
    print(f"[OK] Project '{config.name}' created successfully!")
    print(f"{'='*60}")
    
    if config.dataset:
        print(f"\nDataset: {config.dataset}")
        print(f"Target: {config.target or 'auto-detect'}")
        print(f"Backend: {config.backend}")
        print(f"Problem: {config.problem_type}")
    
    print(f"\nProject location: {Path(config.name).absolute()}")
    print(f"\nNext steps:")
    print(f"  1. cd {config.name}")
    
    if not config.skip_venv:
        print(f"  2. Activate the virtual environment")
        print(f"     Windows: .venv\\Scripts\\activate")
        print(f"     Linux/Mac: source .venv/bin/activate")
    
    if config.dataset:
        print(f"  3. Review the generated pipeline in pipelines/generated/")
        print(f"  4. Open notebooks/starter_notebook.ipynb to begin experimentation")
    else:
        print(f"  2. Add your dataset and run: python -m bukka run --name {config.name} --dataset <path>")


def create_config_template(args: argparse.Namespace) -> None:
    """Create a YAML configuration template file.
    
    Args:
        args: Parsed command-line arguments containing output path.
    """
    output_path = args.output or "bukka_config.yaml"
    
    try:
        created_path = ConfigManager.create_template(output_path)
        logger.info(f"Configuration template created: {created_path}", format_level="h3")
        print(f"[OK] Configuration template created at: {created_path}")
        print(f"\nEdit this file and run: python -m bukka run --config {created_path}")
    except Exception as e:
        logger.error(f"Failed to create config template: {e}")
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


def validate_config(config: BukkaConfig) -> None:
    """Validate configuration and exit if errors found.
    
    Args:
        config: Configuration to validate.
        
    Raises:
        SystemExit: If validation errors are found.
    
    Examples:
        >>> config = BukkaConfig(name="test")
        >>> validate_config(config)  # Should not raise if valid
    """
    errors = config.validate()
    
    if errors:
        logger.error("Validation errors found:")
        for error in errors:
            logger.error(f"  - {error}")
            print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)


def run_project(args: argparse.Namespace) -> None:
    """Run the main project creation workflow.
    
    Args:
        args: Parsed command-line arguments.
    """
    # Load and merge configuration
    try:
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
        
        config_dict = ConfigManager.merge_args_and_config(args, args.config if args.config else None)
        config = BukkaConfig.from_args_and_config(args, config_dict)
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        print(f"[ERROR] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate configuration
    validate_config(config)
    
    # Log project info
    logger.info("Creating Bukka project!", format_level="h1")
    logger.info(f"Project: {config.name}")
    logger.info(f"Dataset: {config.dataset or 'None (will be added later)'}")
    logger.info(f"Target: {config.target or 'None (clustering or to be determined)'}")
    logger.info(f"Backend: {config.backend}")
    logger.info(f"Problem Type: {config.problem_type}")
    
    # Create and run project
    try:
        proj = Project(**config.to_project_kwargs())
        proj.run()
        print_project_success(config)
        
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        print(f"\n[ERROR] Error creating project: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)



def main() -> None:
    """Main CLI entrypoint with subcommands."""
    parser = argparse.ArgumentParser(
        prog='bukka',
        description='Bukka - Machine Learning Project Scaffolding Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a config template
  python -m bukka init-config
  python -m bukka init-config --output my_config.yaml
  
  # Create a project with inline arguments
  python -m bukka run --name my_project --dataset data.csv --target price
  
  # Create a project from config file
  python -m bukka run --config bukka_config.yaml
  
  # Specify backend and problem type
  python -m bukka run -n my_proj -d data.csv -t label --backend pandas --problem-type regression
  
  # Create project structure only (no dataset yet)
  python -m bukka run --name my_project --skip-venv

For more information, visit: https://github.com/pjachim/Bukka
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Subcommand: init-config
    config_parser = subparsers.add_parser(
        'init-config',
        help='Create a YAML configuration template',
        description='Generate a YAML configuration file with default values and documentation.'
    )
    config_parser.add_argument(
        '--output', '-o',
        type=str,
        default='bukka_config.yaml',
        help='Output path for the config template (default: bukka_config.yaml)'
    )
    
    # Subcommand: run
    run_parser = subparsers.add_parser(
        'run',
        help='Create and set up a Bukka project',
        description='Create a new Bukka ML project with automatic pipeline generation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration source
    config_group = run_parser.add_argument_group('configuration')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file (overrides other arguments)'
    )
    
    # Project settings
    project_group = run_parser.add_argument_group('project settings')
    project_group.add_argument(
        '--name', '-n',
        type=str,
        help='Project name / directory to create (required unless using --config)'
    )
    project_group.add_argument(
        '--dataset', '-d',
        type=str,
        help='Path to dataset file (CSV, Parquet, etc.)'
    )
    project_group.add_argument(
        '--target', '-t',
        type=str,
        help='Name of the target column (omit for clustering)'
    )
    project_group.add_argument(
        '--skip-venv', '-sv',
        action='store_true',
        help='Skip virtual environment creation'
    )
    
    # Data processing settings
    data_group = run_parser.add_argument_group('data processing')
    data_group.add_argument(
        '--backend', '-b',
        type=str,
        choices=SUPPORTED_BACKENDS,
        default=None,
        help=f'Dataframe backend (default: polars). Supported: {', '.join(SUPPORTED_BACKENDS)}'
    )
    data_group.add_argument(
        '--train-size',
        type=float,
        default=None,
        help='Train/test split ratio (default: 0.8)'
    )
    data_group.add_argument(
        '--stratify',
        action='store_true',
        default=True,
        help='Stratify train/test split (default: True)'
    )
    data_group.add_argument(
        '--no-stratify',
        dest='stratify',
        action='store_false',
        help='Disable stratified splitting'
    )
    data_group.add_argument(
        '--strata',
        type=str,
        nargs='+',
        help='Column(s) to use for stratification'
    )
    
    # Problem specification
    problem_group = run_parser.add_argument_group('problem specification')
    problem_group.add_argument(
        '--problem-type', '-p',
        type=str,
        choices=PROBLEM_TYPES,
        default=None,
        help=f'ML problem type (default: auto). Options: {', '.join(PROBLEM_TYPES)}'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'init-config':
        create_config_template(args)
    elif args.command == 'run':
        # Validate that either config or name is provided
        if not args.config and not args.name:
            run_parser.error("either --config or --name is required")
        run_project(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
