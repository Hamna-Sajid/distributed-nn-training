# coordination/adaptive_aggregator.py
import numpy as np

class AdaptiveAggregator:
    """
    Combines gradients from multiple workers using adaptive weighting.
    
    Strategy: batch-size weighting (inherited from M1) PLUS two adaptive
    adjustments:
    
    1. Gradient norm clipping: if a worker's gradients are unusually large
       (e.g. due to a bad minibatch), reduce its weight so it doesn't
       dominate the update.
       
    2. Staleness penalty: if a worker arrived very late to the barrier,
       discount its contribution slightly. This is the foundation for
       async training in future work.
    """

    def __init__(self, norm_clip_threshold=10.0, staleness_penalty=0.5):
        """
        Args:
            norm_clip_threshold: if a worker's gradient L2 norm exceeds this
                multiple of the median norm across workers, clip its weight.
            staleness_penalty: factor (0–1) applied to workers that arrived
                more than 2x the median arrival time. 1.0 = no penalty.
        """
        self.norm_clip_threshold = norm_clip_threshold
        self.staleness_penalty = staleness_penalty

        # Track history for reporting
        self.history = []  # list of dicts, one per epoch

    def aggregate(self, gradients, batch_sizes, losses, arrival_times):
        """
        Compute the aggregated gradient update.
        
        Args:
            gradients: dict {worker_id: {layer_name: {'weights': ndarray, 'biases': ndarray}}}
            batch_sizes: dict {worker_id: int}
            losses: dict {worker_id: float}
            arrival_times: dict {worker_id: float} (seconds from barrier open)
        
        Returns:
            aggregated_gradients: dict {layer_name: {'weights': ndarray, 'biases': ndarray}}
            worker_weights: dict {worker_id: float} (for logging/debugging)
        """
        # Filter out failed workers (None gradients)
        valid_workers = [wid for wid, g in gradients.items() if g is not None]
        
        if not valid_workers:
            raise RuntimeError("No valid gradients received from any worker!")

        # --- Step 1: Base weight = batch size fraction ---
        total_samples = sum(batch_sizes[wid] for wid in valid_workers)
        base_weights = {
            wid: batch_sizes[wid] / total_samples
            for wid in valid_workers
        }

        # --- Step 2: Compute per-worker gradient L2 norms ---
        def grad_norm(grad_dict):
            # Flatten all gradient arrays and compute overall L2 norm
            all_vals = []
            for layer in grad_dict.values():
                all_vals.append(layer['weights'].flatten())
                all_vals.append(layer['biases'].flatten())
            return np.linalg.norm(np.concatenate(all_vals))

        norms = {wid: grad_norm(gradients[wid]) for wid in valid_workers}
        median_norm = np.median(list(norms.values()))

        # --- Step 3: Staleness adjustment ---
        if len(arrival_times) > 1:
            median_arrival = np.median([arrival_times[wid] for wid in valid_workers])
        else:
            median_arrival = 0.0

        # --- Step 4: Compute final adaptive weights ---
        adaptive_weights = {}
        for wid in valid_workers:
            w = base_weights[wid]

            # Norm penalty: if this worker's gradient norm is much larger
            # than median, its update may be noisy — reduce its influence
            if median_norm > 0 and norms[wid] > self.norm_clip_threshold * median_norm:
                norm_factor = (self.norm_clip_threshold * median_norm) / norms[wid]
                w *= norm_factor
                print(f"[Aggregator] Worker {wid}: norm {norms[wid]:.2f} >> "
                      f"median {median_norm:.2f}, applying norm clip factor {norm_factor:.3f}")

            # Staleness penalty: if worker arrived much later than median
            if arrival_times.get(wid, 0) > 2.0 * median_arrival and median_arrival > 0.01:
                w *= self.staleness_penalty
                print(f"[Aggregator] Worker {wid}: late arrival "
                      f"({arrival_times[wid]:.2f}s vs median {median_arrival:.2f}s), "
                      f"applying staleness penalty {self.staleness_penalty}")

            adaptive_weights[wid] = w

        # Renormalize so weights sum to 1
        weight_sum = sum(adaptive_weights.values())
        adaptive_weights = {wid: w / weight_sum for wid, w in adaptive_weights.items()}

        # --- Step 5: Weighted sum of gradients ---
        # Use the first valid worker's gradient structure as a template
        first = gradients[valid_workers[0]]
        aggregated = {
            layer: {
                'weights': np.zeros_like(first[layer]['weights']),
                'biases': np.zeros_like(first[layer]['biases'])
            }
            for layer in first.keys()
        }

        for wid in valid_workers:
            w = adaptive_weights[wid]
            for layer in aggregated.keys():
                aggregated[layer]['weights'] += w * gradients[wid][layer]['weights']
                aggregated[layer]['biases']  += w * gradients[wid][layer]['biases']

        # Log this epoch's stats
        avg_loss = sum(losses[wid] * base_weights[wid] for wid in valid_workers)
        self.history.append({
            'worker_weights': adaptive_weights,
            'gradient_norms': norms,
            'avg_loss': avg_loss,
            'active_workers': len(valid_workers),
        })

        return aggregated, adaptive_weights