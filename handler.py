import os
import sys
import time
import logging
from typing import Dict, Any, List

import runpod
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CUDA at module load time (before any handler calls)
# This prevents race conditions in RunPod containers
def _init_cuda():
    """Initialize CUDA with retry logic for RunPod containers."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Just check availability - don't create contexts for each GPU
            # Worker threads will create their own contexts
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"CUDA initialized successfully with {device_count} GPUs")
                return True
            else:
                logger.warning(f"CUDA not available on attempt {attempt + 1}/{max_retries}")
                time.sleep(1)
        except Exception as e:
            logger.warning(f"CUDA init error on attempt {attempt + 1}: {e}")
            time.sleep(1)
    logger.error("Failed to initialize CUDA after all retries")
    return False

# Run CUDA initialization
_cuda_ready = _init_cuda()

from pow.compute.gpu_group import create_gpu_groups, GpuGroup
from pow.compute.autobs_v2 import get_batch_size_for_gpu_group
from pow.compute.worker import ParallelWorkerManager
from pow.models.utils import Params

# Maximum job duration: 7 minutes
MAX_JOB_DURATION = 7 * 60


def handler(event: Dict[str, Any]):
    """
    Parallel streaming nonce generator using multiple GPU groups.

    Each GPU group runs as an independent worker process,
    processing different nonce ranges in parallel.

    Stops when:
    1. Client calls POST /cancel/{job_id}
    2. Timeout after 7 minutes (MAX_JOB_DURATION)

    Input from client (ALL REQUIRED):
    {
        "block_hash": str,
        "block_height": int,
        "public_key": str,
        "r_target": float,
        "batch_size": int,  # This is now total batch size, will be distributed
        "start_nonce": int,
        "params": dict,
    }

    Yields:
    {
        "nonces": [...],
        "dist": [...],
        "batch_number": int,
        "worker_id": int,
        "elapsed_seconds": int,
        ...
    }
    """
    aggregated_batch_count = 0
    total_computed = 0
    total_valid = 0

    try:
        input_data = event.get("input", {})

        # Get ALL parameters from client - NO DEFAULTS
        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        r_target = input_data["r_target"]
        client_batch_size = input_data["batch_size"]
        start_nonce = input_data["start_nonce"]
        params_dict = input_data["params"]

        params = Params(**params_dict)

        # Check CUDA initialization
        if not _cuda_ready:
            raise RuntimeError("CUDA initialization failed - no GPU support available")

        # Auto-detect GPUs and create groups
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPUs")

        # Create GPU groups based on VRAM requirements
        gpu_groups = create_gpu_groups(params=params)
        n_workers = len(gpu_groups)

        logger.info(f"Created {n_workers} GPU groups for parallel processing:")
        for i, group in enumerate(gpu_groups):
            logger.info(f"  Worker {i}: {group} (VRAM: {group.get_total_vram_gb():.1f}GB)")

        # Calculate batch size per worker
        # Use auto-calculated batch size based on GPU memory
        batch_sizes = []
        for group in gpu_groups:
            bs = get_batch_size_for_gpu_group(group, params)
            batch_sizes.append(bs)
            logger.info(f"  Worker batch size for {group.devices}: {bs}")

        # Use minimum batch size across all groups to ensure consistency
        batch_size_per_worker = min(batch_sizes)
        total_batch_size = batch_size_per_worker * n_workers

        logger.info(f"START: block={block_height}, workers={n_workers}, "
                   f"batch_per_worker={batch_size_per_worker}, total_batch={total_batch_size}, "
                   f"start={start_nonce}")

        # Convert GPU groups to device string lists
        gpu_group_devices = [group.get_device_strings() for group in gpu_groups]

        # Create and start parallel worker manager
        manager = ParallelWorkerManager(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            r_target=r_target,
            batch_size_per_worker=batch_size_per_worker,
            gpu_groups=gpu_group_devices,
            start_nonce=start_nonce,
            max_duration=MAX_JOB_DURATION,
        )

        manager.start()

        # Wait for all workers to initialize models
        if not manager.wait_for_ready(timeout=180):
            logger.error("Workers failed to initialize within timeout")
            yield {
                "error": "Worker initialization timeout",
                "error_type": "TimeoutError",
            }
            manager.stop()
            return

        logger.info("All workers ready, starting streaming")

        start_time = time.time()
        last_result_time = start_time

        # Streaming results from all workers
        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > MAX_JOB_DURATION:
                logger.info(f"TIMEOUT: {elapsed:.0f}s exceeded {MAX_JOB_DURATION}s limit")
                break

            # Check if all workers have stopped
            if not manager.is_alive():
                logger.info("All workers have stopped")
                break

            # Get results from workers
            results = manager.get_results(timeout=0.5)

            if not results:
                # No results available, check for stall
                if time.time() - last_result_time > 60:
                    logger.warning("No results for 60s, workers may be stuck")
                continue

            last_result_time = time.time()

            # Yield each result
            for result in results:
                if "error" in result:
                    logger.error(f"Worker {result.get('worker_id')} error: {result['error']}")
                    yield result
                    continue

                aggregated_batch_count += 1
                total_computed += result.get("batch_computed", 0)
                total_valid += result.get("batch_valid", 0)

                # Add aggregated stats and fix batch_number for deduplication
                result["aggregated_batch_number"] = aggregated_batch_count
                result["aggregated_total_computed"] = total_computed
                result["aggregated_total_valid"] = total_valid
                result["n_workers"] = n_workers
                # Override batch_number with globally unique value
                # (delegation_controller deduplicates by batch_number)
                result["batch_number"] = aggregated_batch_count

                logger.info(f"Batch #{aggregated_batch_count} from worker {result['worker_id']}: "
                           f"{result.get('batch_valid', 0)} valid, elapsed={int(elapsed)}s")

                yield result

        logger.info(f"STOPPED: {aggregated_batch_count} batches, {total_computed} computed, {total_valid} valid")
        manager.stop()

    except GeneratorExit:
        logger.info(f"CANCELLED: {aggregated_batch_count} batches, {total_computed} computed, {total_valid} valid")
        try:
            manager.stop()
        except:
            pass
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        yield {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        try:
            manager.stop()
        except:
            pass


# Start serverless handler with streaming support
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
