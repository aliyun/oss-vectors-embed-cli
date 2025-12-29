"""OSS configuration utilities."""

import alibabacloud_oss_v2 as oss
from oss_vectors.__version__ import __version__

def setup_oss_cfg(account_id=None, region=None):

    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    if not region:
        region = get_region()
    cfg.region = region
    cfg.account_id = account_id
    return cfg


def get_region(region=None):
    if region:
        return region

    # Default to cn-hangzhou
    return 'cn-hangzhou'


def get_user_agent():
    """
    Get the CLI user agent string for logging/debugging.

    Returns:
        User agent string for display
    """
    return f"oss-vectors-embed-cli/{__version__}"
