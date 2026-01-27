"""Tests for retry logic with exponential backoff."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from evalhub.client.base import (
    BaseAsyncClient,
    BaseSyncClient,
    _calculate_retry_delay,
)


class TestRetryDelayCalculation:
    """Test retry delay calculation with exponential backoff."""

    def test_exponential_backoff_without_randomization(self) -> None:
        """Test exponential backoff calculation without randomization."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_max_delay=60.0,
            retry_randomization=False,
        )

        # First retry: 1.0 * 2^0 = 1.0
        assert (
            _calculate_retry_delay(
                0,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 1.0
        )

        # Second retry: 1.0 * 2^1 = 2.0
        assert (
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 2.0
        )

        # Third retry: 1.0 * 2^2 = 4.0
        assert (
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 4.0
        )

        # Fourth retry: 1.0 * 2^3 = 8.0
        assert (
            _calculate_retry_delay(
                3,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 8.0
        )

    def test_exponential_backoff_with_max_delay(self) -> None:
        """Test that backoff respects max_delay."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_max_delay=5.0,
            retry_randomization=False,
        )

        # Should cap at max_delay
        assert (
            _calculate_retry_delay(
                0,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 1.0
        )
        assert (
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 2.0
        )
        assert (
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 4.0
        )
        assert (
            _calculate_retry_delay(
                3,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 5.0
        )  # Capped
        assert (
            _calculate_retry_delay(
                4,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 5.0
        )  # Capped
        assert (
            _calculate_retry_delay(
                10,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 5.0
        )  # Capped

    def test_exponential_backoff_with_randomization(self) -> None:
        """Test that randomization adds jitter to delays."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_max_delay=60.0,
            retry_randomization=True,
        )

        # With randomization, delays should be between 50% and 100% of calculated value
        for attempt in range(5):
            expected_base = min(1.0 * (2.0**attempt), 60.0)
            delay = _calculate_retry_delay(
                attempt,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            assert expected_base * 0.5 <= delay <= expected_base

    def test_custom_backoff_factor(self) -> None:
        """Test custom backoff factor."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=3.0,
            retry_max_delay=100.0,
            retry_randomization=False,
        )

        # With factor 3.0: 1.0, 3.0, 9.0, 27.0, 81.0
        assert (
            _calculate_retry_delay(
                0,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 1.0
        )
        assert (
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 3.0
        )
        assert (
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 9.0
        )
        assert (
            _calculate_retry_delay(
                3,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 27.0
        )
        assert (
            _calculate_retry_delay(
                4,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 81.0
        )

    def test_exponential_growth_progression(self) -> None:
        """Test that delays actually grow exponentially, not linearly."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_max_delay=1000.0,
            retry_randomization=False,
        )

        delays = [
            _calculate_retry_delay(
                i,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            for i in range(6)
        ]
        # Expected: [1, 2, 4, 8, 16, 32]

        # Verify exponential growth: each delay is double the previous
        for i in range(1, len(delays)):
            assert delays[i] == delays[i - 1] * 2.0

        # Verify it's NOT linear (where increments would be constant)
        # In linear: [1, 2, 3, 4, 5, 6] - increments of 1
        # In exponential: [1, 2, 4, 8, 16, 32] - increments grow
        increments = [delays[i] - delays[i - 1] for i in range(1, len(delays))]
        # Increments should be increasing: [1, 2, 4, 8, 16]
        for i in range(1, len(increments)):
            assert increments[i] > increments[i - 1]

    def test_fractional_backoff_factor(self) -> None:
        """Test backoff with fractional multiplier."""
        client = BaseAsyncClient(
            retry_initial_delay=2.0,
            retry_backoff_factor=1.5,
            retry_max_delay=50.0,
            retry_randomization=False,
        )

        # With factor 1.5: 2.0, 3.0, 4.5, 6.75, 10.125, 15.1875
        assert (
            _calculate_retry_delay(
                0,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 2.0
        )
        assert (
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 3.0
        )
        assert (
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 4.5
        )
        assert (
            _calculate_retry_delay(
                3,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 6.75
        )
        assert (
            abs(
                _calculate_retry_delay(
                    4,
                    client.retry_initial_delay,
                    client.retry_max_delay,
                    client.retry_backoff_factor,
                    client.retry_randomization,
                )
                - 10.125
            )
            < 0.001
        )

    def test_very_small_initial_delay(self) -> None:
        """Test with very small initial delay."""
        client = BaseAsyncClient(
            retry_initial_delay=0.001,
            retry_backoff_factor=10.0,
            retry_max_delay=10.0,
            retry_randomization=False,
        )

        # 0.001, 0.01, 0.1, 1.0, 10.0 (capped)
        assert (
            _calculate_retry_delay(
                0,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 0.001
        )
        assert (
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 0.01
        )
        assert (
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 0.1
        )
        assert (
            _calculate_retry_delay(
                3,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 1.0
        )
        assert (
            _calculate_retry_delay(
                4,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 10.0
        )  # Capped

    def test_very_large_backoff_factor(self) -> None:
        """Test with large backoff factor quickly reaches max."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=10.0,
            retry_max_delay=100.0,
            retry_randomization=False,
        )

        # 1.0, 10.0, 100.0 (capped), 100.0 (capped)
        assert (
            _calculate_retry_delay(
                0,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 1.0
        )
        assert (
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 10.0
        )
        assert (
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 100.0
        )
        assert (
            _calculate_retry_delay(
                3,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 100.0
        )  # Already capped

    def test_backoff_factor_of_one(self) -> None:
        """Test backoff factor of 1.0 results in constant delay (linear)."""
        client = BaseAsyncClient(
            retry_initial_delay=5.0,
            retry_backoff_factor=1.0,
            retry_max_delay=100.0,
            retry_randomization=False,
        )

        # With factor 1.0, all delays should be the same
        for attempt in range(10):
            assert (
                _calculate_retry_delay(
                    attempt,
                    client.retry_initial_delay,
                    client.retry_max_delay,
                    client.retry_backoff_factor,
                    client.retry_randomization,
                )
                == 5.0
            )

    def test_high_attempt_numbers(self) -> None:
        """Test that very high attempt numbers don't cause overflow or errors."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_max_delay=3600.0,  # 1 hour max
            retry_randomization=False,
        )

        # Attempt 20: 1.0 * 2^20 = 1,048,576 seconds (but capped at 3600)
        delay = _calculate_retry_delay(
            20,
            client.retry_initial_delay,
            client.retry_max_delay,
            client.retry_backoff_factor,
            client.retry_randomization,
        )
        assert delay == 3600.0

        # Attempt 100: Even higher, but still capped
        delay = _calculate_retry_delay(
            100,
            client.retry_initial_delay,
            client.retry_max_delay,
            client.retry_backoff_factor,
            client.retry_randomization,
        )
        assert delay == 3600.0

    def test_zero_initial_delay(self) -> None:
        """Test with zero initial delay."""
        client = BaseAsyncClient(
            retry_initial_delay=0.0,
            retry_backoff_factor=2.0,
            retry_max_delay=60.0,
            retry_randomization=False,
        )

        # All delays should be 0.0
        for attempt in range(10):
            assert (
                _calculate_retry_delay(
                    attempt,
                    client.retry_initial_delay,
                    client.retry_max_delay,
                    client.retry_backoff_factor,
                    client.retry_randomization,
                )
                == 0.0
            )

    def test_randomization_produces_different_values(self) -> None:
        """Test that randomization produces different values on repeated calls."""
        client = BaseAsyncClient(
            retry_initial_delay=10.0,
            retry_backoff_factor=2.0,
            retry_max_delay=1000.0,
            retry_randomization=True,
        )

        # Call many times and collect results
        delays = [
            _calculate_retry_delay(
                3,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            for _ in range(100)
        ]

        # All should be in valid range (expected base: 10.0 * 2^3 = 80.0)
        for delay in delays:
            assert 40.0 <= delay <= 80.0

        # Should have variation (not all the same)
        unique_delays = len(set(delays))
        assert unique_delays > 50  # Should have significant variation

    def test_randomization_statistical_properties(self) -> None:
        """Test that randomization follows expected distribution."""
        client = BaseAsyncClient(
            retry_initial_delay=10.0,
            retry_backoff_factor=2.0,
            retry_max_delay=1000.0,
            retry_randomization=True,
        )

        # Expected base for attempt 2: 10.0 * 2^2 = 40.0
        # Randomization range: 20.0 to 40.0
        delays = [
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            for _ in range(1000)
        ]

        # All should be in range
        assert all(20.0 <= d <= 40.0 for d in delays)

        # Statistical properties
        avg_delay = sum(delays) / len(delays)
        # Average should be around 75% of max (30.0) with some tolerance
        assert 28.0 <= avg_delay <= 32.0

    def test_backoff_with_different_initial_delays(self) -> None:
        """Test that different initial delays scale appropriately."""
        configs = [
            (0.5, 2.0, 100.0),
            (1.0, 2.0, 100.0),
            (2.0, 2.0, 100.0),
            (5.0, 2.0, 100.0),
        ]

        for initial_delay, factor, max_delay in configs:
            client = BaseAsyncClient(
                retry_initial_delay=initial_delay,
                retry_backoff_factor=factor,
                retry_max_delay=max_delay,
                retry_randomization=False,
            )

            # Verify the progression scales with initial delay
            for attempt in range(5):
                expected = min(initial_delay * (factor**attempt), max_delay)
                actual = _calculate_retry_delay(
                    attempt,
                    client.retry_initial_delay,
                    client.retry_max_delay,
                    client.retry_backoff_factor,
                    client.retry_randomization,
                )
                assert abs(actual - expected) < 0.001

    def test_sync_client_has_same_calculation(self) -> None:
        """Test that sync client uses identical delay calculation."""
        async_client = BaseAsyncClient(
            retry_initial_delay=2.0,
            retry_backoff_factor=3.0,
            retry_max_delay=100.0,
            retry_randomization=False,
        )

        sync_client = BaseSyncClient(
            retry_initial_delay=2.0,
            retry_backoff_factor=3.0,
            retry_max_delay=100.0,
            retry_randomization=False,
        )

        # Both should produce identical delays
        for attempt in range(10):
            async_delay = _calculate_retry_delay(
                attempt,
                async_client.retry_initial_delay,
                async_client.retry_max_delay,
                async_client.retry_backoff_factor,
                async_client.retry_randomization,
            )
            sync_delay = _calculate_retry_delay(
                attempt,
                sync_client.retry_initial_delay,
                sync_client.retry_max_delay,
                sync_client.retry_backoff_factor,
                sync_client.retry_randomization,
            )
            assert async_delay == sync_delay

        sync_client.close()


class TestAsyncClientRetry:
    """Test async client retry logic."""

    @pytest.mark.asyncio
    async def test_successful_request_no_retry(self) -> None:
        """Test successful request doesn't trigger retries."""
        client = BaseAsyncClient(max_retries=3)

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            response = await client._request("GET", "/test")

            # Should only be called once (no retries)
            assert mock_request.call_count == 1
            assert response == mock_response

        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self) -> None:
        """Test retry on timeout error."""
        client = BaseAsyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            # First two calls timeout, third succeeds
            mock_request.side_effect = [
                httpx.TimeoutException("Timeout 1"),
                httpx.TimeoutException("Timeout 2"),
                mock_response,
            ]

            start_time = time.time()
            response = await client._request("GET", "/test")
            elapsed = time.time() - start_time

            # Should have retried twice
            assert mock_request.call_count == 3
            assert response == mock_response

            # Should have delayed for retries (0.01s + 0.02s = 0.03s minimum)
            assert elapsed >= 0.03

        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self) -> None:
        """Test retry on 5xx server errors."""
        client = BaseAsyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        mock_response_500 = Mock(spec=httpx.Response)
        mock_response_500.status_code = 500

        mock_response_200 = Mock(spec=httpx.Response)
        mock_response_200.status_code = 200

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            # First two calls return 500, third succeeds
            mock_request.side_effect = [
                httpx.HTTPStatusError(
                    "Server error", request=Mock(), response=mock_response_500
                ),
                httpx.HTTPStatusError(
                    "Server error", request=Mock(), response=mock_response_500
                ),
                mock_response_200,
            ]

            response = await client._request("GET", "/test")

            assert mock_request.call_count == 3
            assert response == mock_response_200

        await client.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self) -> None:
        """Test no retry on 4xx client errors."""
        client = BaseAsyncClient(max_retries=3)

        mock_response_404 = Mock(spec=httpx.Response)
        mock_response_404.status_code = 404

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = httpx.HTTPStatusError(
                "Not found", request=Mock(), response=mock_response_404
            )

            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/test")

            # Should only be called once (no retries for 4xx)
            assert mock_request.call_count == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Test that max retries is respected."""
        client = BaseAsyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            # Always timeout
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(httpx.TimeoutException):
                await client._request("GET", "/test")

            # Should be called max_retries + 1 times (initial + 2 retries)
            assert mock_request.call_count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self) -> None:
        """Test retry on connection errors."""
        client = BaseAsyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            # First two calls fail with connection error, third succeeds
            mock_request.side_effect = [
                httpx.ConnectError("Connection failed"),
                httpx.ConnectError("Connection failed"),
                mock_response,
            ]

            response = await client._request("GET", "/test")

            assert mock_request.call_count == 3
            assert response == mock_response

        await client.close()


class TestSyncClientRetry:
    """Test sync client retry logic."""

    def test_successful_request_no_retry(self) -> None:
        """Test successful request doesn't trigger retries."""
        client = BaseSyncClient(max_retries=3)

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        with patch.object(client._client, "request") as mock_request:
            mock_request.return_value = mock_response

            response = client._request("GET", "/test")

            # Should only be called once (no retries)
            assert mock_request.call_count == 1
            assert response == mock_response

        client.close()

    def test_retry_on_timeout(self) -> None:
        """Test retry on timeout error."""
        client = BaseSyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(client._client, "request") as mock_request:
            # First two calls timeout, third succeeds
            mock_request.side_effect = [
                httpx.TimeoutException("Timeout 1"),
                httpx.TimeoutException("Timeout 2"),
                mock_response,
            ]

            start_time = time.time()
            response = client._request("GET", "/test")
            elapsed = time.time() - start_time

            # Should have retried twice
            assert mock_request.call_count == 3
            assert response == mock_response

            # Should have delayed for retries (0.01s + 0.02s = 0.03s minimum)
            assert elapsed >= 0.03

        client.close()

    def test_retry_on_server_error(self) -> None:
        """Test retry on 5xx server errors."""
        client = BaseSyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        mock_response_500 = Mock(spec=httpx.Response)
        mock_response_500.status_code = 500

        mock_response_200 = Mock(spec=httpx.Response)
        mock_response_200.status_code = 200

        with patch.object(client._client, "request") as mock_request:
            # First two calls return 500, third succeeds
            mock_request.side_effect = [
                httpx.HTTPStatusError(
                    "Server error", request=Mock(), response=mock_response_500
                ),
                httpx.HTTPStatusError(
                    "Server error", request=Mock(), response=mock_response_500
                ),
                mock_response_200,
            ]

            response = client._request("GET", "/test")

            assert mock_request.call_count == 3
            assert response == mock_response_200

        client.close()

    def test_no_retry_on_client_error(self) -> None:
        """Test no retry on 4xx client errors."""
        client = BaseSyncClient(max_retries=3)

        mock_response_404 = Mock(spec=httpx.Response)
        mock_response_404.status_code = 404

        with patch.object(client._client, "request") as mock_request:
            mock_request.side_effect = httpx.HTTPStatusError(
                "Not found", request=Mock(), response=mock_response_404
            )

            with pytest.raises(httpx.HTTPStatusError):
                client._request("GET", "/test")

            # Should only be called once (no retries for 4xx)
            assert mock_request.call_count == 1

        client.close()

    def test_max_retries_exceeded(self) -> None:
        """Test that max retries is respected."""
        client = BaseSyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        with patch.object(client._client, "request") as mock_request:
            # Always timeout
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(httpx.TimeoutException):
                client._request("GET", "/test")

            # Should be called max_retries + 1 times (initial + 2 retries)
            assert mock_request.call_count == 3

        client.close()

    def test_retry_on_connection_error(self) -> None:
        """Test retry on connection errors."""
        client = BaseSyncClient(
            max_retries=2, retry_initial_delay=0.01, retry_randomization=False
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(client._client, "request") as mock_request:
            # First two calls fail with connection error, third succeeds
            mock_request.side_effect = [
                httpx.ConnectError("Connection failed"),
                httpx.ConnectError("Connection failed"),
                mock_response,
            ]

            response = client._request("GET", "/test")

            assert mock_request.call_count == 3
            assert response == mock_response

        client.close()


class TestExponentialBackoffTiming:
    """Test actual timing behavior of exponential backoff during retries."""

    @pytest.mark.asyncio
    async def test_async_retry_timing_follows_exponential_backoff(self) -> None:
        """Test that async retries actually wait with exponential delays."""
        client = BaseAsyncClient(
            max_retries=3,
            retry_initial_delay=0.1,
            retry_backoff_factor=2.0,
            retry_max_delay=10.0,
            retry_randomization=False,
        )

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            # Always fail with timeout
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            start_time = time.time()
            try:
                await client._request("GET", "/test")
            except httpx.TimeoutException:
                pass

            elapsed = time.time() - start_time

            # Expected delays: 0.1s, 0.2s, 0.4s = 0.7s total
            # Allow some tolerance for execution overhead
            assert 0.65 <= elapsed <= 0.85

            # Should have made 4 attempts (initial + 3 retries)
            assert mock_request.call_count == 4

        await client.close()

    def test_sync_retry_timing_follows_exponential_backoff(self) -> None:
        """Test that sync retries actually wait with exponential delays."""
        client = BaseSyncClient(
            max_retries=3,
            retry_initial_delay=0.1,
            retry_backoff_factor=2.0,
            retry_max_delay=10.0,
            retry_randomization=False,
        )

        with patch.object(client._client, "request") as mock_request:
            # Always fail with timeout
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            start_time = time.time()
            try:
                client._request("GET", "/test")
            except httpx.TimeoutException:
                pass

            elapsed = time.time() - start_time

            # Expected delays: 0.1s, 0.2s, 0.4s = 0.7s total
            # Allow some tolerance for execution overhead
            assert 0.65 <= elapsed <= 0.85

            # Should have made 4 attempts (initial + 3 retries)
            assert mock_request.call_count == 4

        client.close()

    @pytest.mark.asyncio
    async def test_timing_with_randomization_is_variable(self) -> None:
        """Test that randomization causes variable retry timing."""
        timings = []

        for _ in range(5):
            client = BaseAsyncClient(
                max_retries=2,
                retry_initial_delay=0.1,
                retry_backoff_factor=2.0,
                retry_max_delay=10.0,
                retry_randomization=True,
            )

            with patch.object(
                client._client, "request", new_callable=AsyncMock
            ) as mock_request:
                mock_request.side_effect = httpx.TimeoutException("Timeout")

                start_time = time.time()
                try:
                    await client._request("GET", "/test")
                except httpx.TimeoutException:
                    pass

                elapsed = time.time() - start_time
                timings.append(elapsed)

            await client.close()

        # With randomization, timings should vary
        # Expected range: 0.15s (50% of 0.3s total) to 0.3s
        # But they should not all be identical
        assert len(set(f"{t:.3f}" for t in timings)) > 1  # Not all the same

    @pytest.mark.asyncio
    async def test_max_delay_limits_total_wait_time(self) -> None:
        """Test that max_delay prevents excessively long waits."""
        client = BaseAsyncClient(
            max_retries=5,
            retry_initial_delay=1.0,
            retry_backoff_factor=10.0,  # Very aggressive
            retry_max_delay=0.5,  # But capped low
            retry_randomization=False,
        )

        with patch.object(
            client._client, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            start_time = time.time()
            try:
                await client._request("GET", "/test")
            except httpx.TimeoutException:
                pass

            elapsed = time.time() - start_time

            # With aggressive backoff but low max_delay:
            # Delays: 0.5 (capped), 0.5, 0.5, 0.5, 0.5 = 2.5s total
            # Allow tolerance
            assert 2.4 <= elapsed <= 2.7

        await client.close()

    def test_total_time_calculation(self) -> None:
        """Test calculating total possible retry time."""
        client = BaseSyncClient(
            max_retries=3,
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_max_delay=60.0,
            retry_randomization=False,
        )

        # Calculate total potential delay
        total_delay = sum(
            _calculate_retry_delay(
                i,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            for i in range(client.max_retries)
        )

        # Expected: 1.0 + 2.0 + 4.0 = 7.0 seconds
        assert total_delay == 7.0

        client.close()


class TestRetryConfiguration:
    """Test different retry configurations."""

    @pytest.mark.asyncio
    async def test_custom_retry_parameters(self) -> None:
        """Test custom retry parameters."""
        client = BaseAsyncClient(
            max_retries=5,
            retry_initial_delay=0.5,
            retry_max_delay=30.0,
            retry_backoff_factor=1.5,
            retry_randomization=False,
        )

        assert client.max_retries == 5
        assert client.retry_initial_delay == 0.5
        assert client.retry_max_delay == 30.0
        assert client.retry_backoff_factor == 1.5
        assert client.retry_randomization is False

        # Test delay calculation
        assert (
            _calculate_retry_delay(
                0,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 0.5
        )
        assert (
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 0.75
        )
        assert (
            _calculate_retry_delay(
                2,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            == 1.125
        )

        await client.close()

    def test_zero_retries(self) -> None:
        """Test with zero retries."""
        client = BaseSyncClient(max_retries=0)

        mock_response_500 = Mock(spec=httpx.Response)
        mock_response_500.status_code = 500

        with patch.object(client._client, "request") as mock_request:
            mock_request.side_effect = httpx.HTTPStatusError(
                "Server error", request=Mock(), response=mock_response_500
            )

            with pytest.raises(httpx.HTTPStatusError):
                client._request("GET", "/test")

            # Should only be called once (no retries)
            assert mock_request.call_count == 1

        client.close()

    @pytest.mark.asyncio
    async def test_disable_randomization(self) -> None:
        """Test that disabling randomization produces consistent delays."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_randomization=False,
        )

        # Multiple calls should return same value
        delays = [
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            for _ in range(10)
        ]
        assert all(d == 2.0 for d in delays)

        await client.close()

    @pytest.mark.asyncio
    async def test_enable_randomization_variation(self) -> None:
        """Test that enabling randomization produces variable delays."""
        client = BaseAsyncClient(
            retry_initial_delay=1.0,
            retry_backoff_factor=2.0,
            retry_randomization=True,
        )

        # Multiple calls should return different values
        delays = [
            _calculate_retry_delay(
                1,
                client.retry_initial_delay,
                client.retry_max_delay,
                client.retry_backoff_factor,
                client.retry_randomization,
            )
            for _ in range(100)
        ]
        unique_delays = set(delays)

        # Should have some variation (not all the same)
        assert len(unique_delays) > 1

        # All delays should be in valid range
        for d in delays:
            assert 1.0 <= d <= 2.0

        await client.close()
