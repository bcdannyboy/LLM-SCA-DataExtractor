"""
Target LLM runner for executing SCA attacks.

This module handles interaction with target LLMs, including rate limiting,
retry logic, and response handling.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

from ..models.data_models import Sequence, Response


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    max_concurrent: int = 10


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.request_times: List[float] = []
        self.token_count = 0
        self.token_reset_time = time.time()
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def acquire(self, estimated_tokens: int = 100):
        """
        Acquire permission to make a request.
        
        Args:
            estimated_tokens: Estimated tokens for the request
        """
        async with self.semaphore:
            # Clean old request times
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # Check request rate
            if len(self.request_times) >= self.config.requests_per_minute:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Check token rate
            if current_time - self.token_reset_time >= 60:
                self.token_count = 0
                self.token_reset_time = current_time
            
            if self.token_count + estimated_tokens > self.config.tokens_per_minute:
                sleep_time = 60 - (current_time - self.token_reset_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    self.token_count = 0
                    self.token_reset_time = time.time()
            
            # Record request
            self.request_times.append(time.time())
            self.token_count += estimated_tokens


class TargetLLMRunner(Runnable):
    """
    Executes SCA attacks against target LLMs.
    
    This runner wraps any LangChain ChatModel and provides rate limiting,
    retry logic, and response handling.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        rate_limit_config: Optional[RateLimitConfig] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the runner.
        
        Args:
            model: LangChain chat model to use
            rate_limit_config: Rate limiting configuration
            system_prompt: Optional system prompt
            temperature: Model temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception,))
    )
    async def _invoke_with_retry(self, messages: List[Any]) -> AIMessage:
        """
        Invoke model with retry logic.
        
        Args:
            messages: Messages to send to model
            
        Returns:
            Model response
        """
        try:
            result = await self.model.ainvoke(messages)
            return result
        except Exception as e:
            raise
    
    async def ainvoke(self, sequence: Union[Sequence, str]) -> Response:
        """
        Execute attack with a single sequence.
        
        Args:
            sequence: Attack sequence or string
            
        Returns:
            Response object
        """
        # Handle both Sequence objects and raw strings
        if isinstance(sequence, str):
            sequence = Sequence(content=sequence)
        
        # Estimate tokens (rough approximation)
        estimated_tokens = len(sequence.content.split()) * 2 + self.max_tokens
        
        # Apply rate limiting
        await self.rate_limiter.acquire(estimated_tokens)
        
        # Prepare messages
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=sequence.content))
        
        # Time the request
        start_time = time.time()
        
        try:
            # Invoke model with retry
            ai_message = await self._invoke_with_retry(messages)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create response object
            response = Response(
                sequence_id=sequence.id,
                model=self.model.model_name if hasattr(self.model, 'model_name') else str(type(self.model)),
                content=ai_message.content,
                latency_ms=latency_ms,
                metadata={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "system_prompt_used": bool(self.system_prompt)
                }
            )
            
            # Add token usage if available
            if hasattr(ai_message, 'usage_metadata') and ai_message.usage_metadata is not None:
                try:
                    # Try dictionary-like access first
                    if hasattr(ai_message.usage_metadata, 'get'):
                        response.tokens_used = ai_message.usage_metadata.get('total_tokens')
                    # Try attribute access for UsageMetadata objects
                    elif hasattr(ai_message.usage_metadata, 'total_tokens'):
                        response.tokens_used = ai_message.usage_metadata.total_tokens
                except (AttributeError, KeyError):
                    # If all else fails, just skip token usage
                    pass
            
            return response
            
        except Exception as e:
            # Create error response
            return Response(
                sequence_id=sequence.id,
                model=self.model.model_name if hasattr(self.model, 'model_name') else str(type(self.model)),
                content=f"ERROR: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
    
    def invoke(self, sequence: Union[Sequence, str]) -> Response:
        """
        Synchronous wrapper for ainvoke.
        
        Args:
            sequence: Attack sequence
            
        Returns:
            Response object
        """
        return asyncio.run(self.ainvoke(sequence))
    
    async def abatch(
        self,
        sequences: List[Union[Sequence, str]],
        max_concurrency: Optional[int] = None
    ) -> List[Response]:
        """
        Execute attacks with multiple sequences in parallel.
        
        Args:
            sequences: List of attack sequences
            max_concurrency: Maximum concurrent requests
            
        Returns:
            List of Response objects
        """
        if max_concurrency:
            # Create a new semaphore for batch concurrency
            batch_semaphore = asyncio.Semaphore(max_concurrency)
            
            async def limited_invoke(seq):
                async with batch_semaphore:
                    return await self.ainvoke(seq)
            
            tasks = [limited_invoke(seq) for seq in sequences]
        else:
            tasks = [self.ainvoke(seq) for seq in sequences]
        
        return await asyncio.gather(*tasks)
    
    def batch(
        self,
        sequences: List[Union[Sequence, str]],
        max_concurrency: Optional[int] = None
    ) -> List[Response]:
        """
        Synchronous wrapper for abatch.
        
        Args:
            sequences: List of attack sequences
            max_concurrency: Maximum concurrent requests
            
        Returns:
            List of Response objects
        """
        return asyncio.run(self.abatch(sequences, max_concurrency))
    
    def stream(self, sequence: Union[Sequence, str]):
        """
        Stream response tokens (if supported by model).
        
        Args:
            sequence: Attack sequence
            
        Yields:
            Response tokens
        """
        # This would require model-specific streaming implementation
        raise NotImplementedError("Streaming not yet implemented")