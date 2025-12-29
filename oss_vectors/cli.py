#!/usr/bin/env python3
"""Main CLI entry point for OSS Vectors."""

import click
from rich.console import Console
from rich.traceback import install
from oss_vectors.commands.embed_put import embed_put
from oss_vectors.commands.embed_query import embed_query
from oss_vectors.utils.config import setup_oss_cfg

# Install rich traceback handler
install(show_locals=True)
console = Console()


@click.group()
@click.version_option(version="0.1.0")
@click.option('--account-id', help='Alibaba Cloud account id to use')
@click.option('--vectors-region', help='OSS vectors bucket region name (config defaults)')
@click.option('--vectors-endpoint', help='The domain names that other services can use to access OSS vectors bucket')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed logging')
@click.pass_context
def cli(ctx, account_id, vectors_region, vectors_endpoint, debug):
    """OSS Vectors Embed CLI - Standalone tool for vector embedding operations with OSS and Dashscope."""
    ctx.ensure_object(dict)
    cfg = setup_oss_cfg(account_id, vectors_region)
    if vectors_endpoint is not None:
        cfg.endpoint = vectors_endpoint
    ctx.obj['cfg'] = cfg
    ctx.obj['console'] = console
    ctx.obj['debug'] = debug

    if debug:
        console.print("[dim] Debug mode enabled[/dim]")
        console.print(f"[dim] Alibaba Cloud Profile: {account_id}[/dim]")
        console.print(f"[dim] Alibaba Cloud Vector Bucket Region: {vectors_region}[/dim]")


# Register commands as subcommands
cli.add_command(embed_put, name='put')
cli.add_command(embed_query, name='query')


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise


if __name__ == '__main__':
    main()
