"""
Retry utility for Claude API calls with exponential backoff.

Handles transient failures (429 rate limits, 500 server errors, timeouts)
without changing any agent logic. Both the Macro Strategist's _interpret()
and the Director's chat() use this.
"""

import asyncio
import logging
from typing import Callable, TypeVar

logger = logging.getLogger("finias.utils.retry")

T = TypeVar("T")


async def retry_claude_call(
    call: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
    retry_on: tuple = None,
) -> T:
    """
    Execute a Claude API call with exponential backoff retry.

    Args:
        call: An async callable (the API call to retry)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        retry_on: Tuple of exception types to retry on.
                  Defaults to anthropic rate limit and API status errors.

    Returns:
        The result of the successful call.

    Raises:
        The last exception if all retries fail.
    """
    import anthropic

    if retry_on is None:
        retry_on = (
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        )

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except retry_on as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Claude API call failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Claude API call failed after {max_retries + 1} attempts: "
                    f"{type(e).__name__}: {e}"
                )

    raise last_exception
