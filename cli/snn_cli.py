from __future__ import annotations

import argparse
import os

import numpy as np

from src.train.datasets import SpeechCommandsDataLoader, DEFAULT_KEYWORDS
from src.train.train import (
    TwoLayerModel,
    accuracy,
    build_default_two_layer,
    predict_softmax,
    simulate_get_counts,
    train_softmax,
)
from src.learning.stdp import STDPParameters, STDPState, initialize_stdp_state, stdp_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNN keyword spotting baseline or STDP training")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Speech Commands root directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=50, help="Limit number of train batches for a quick demo")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "stdp"], help="Training mode")
    parser.add_argument("--no_jit", action="store_true", help="Disable numba JIT fast path")
    args = parser.parse_args()

    dl = SpeechCommandsDataLoader(
        root=args.data_root,
        keywords=DEFAULT_KEYWORDS,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        dt=args.dt,
        r_max=100.0,
        n_mels=40,
    )

    # Peek one batch to infer input feature dimension
    first_batch, first_labels = next(iter(dl.iter_split("train")))
    T, B, F = first_batch.shape
    print(f"Inferred features: {F}, classes: {dl.num_classes}")

    model = build_default_two_layer(num_input=F, num_hidden=args.hidden, num_output=dl.num_classes)
    if args.no_jit:
        # Rewire engine to use non-fast functions by monkeypatching if needed
        from src.engine import sim as sim_mod
        from src.models import lif as lif_mod
        from src.models import synapse as syn_mod
        sim_mod.dense_layer_step.__globals__["synapse_step_fast"] = syn_mod.synapse_step
        sim_mod.dense_layer_step.__globals__["lif_step_fast"] = lif_mod.lif_step

    if args.mode == "baseline":
        # Collect counts for a subset of batches for training
        train_features: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        batches_done = 0
        for spikes, labels in dl.iter_split("train"):
            counts1, counts2 = simulate_get_counts(model, spikes, dt=args.dt)
            feats = counts1.astype(np.float64)
            train_features.append(feats)
            train_labels.append(labels)
            batches_done += 1
            if batches_done >= args.max_batches:
                break
        X_train = np.concatenate(train_features, axis=0)
        y_train = np.concatenate(train_labels, axis=0)

        W, b = train_softmax(X_train, y_train, num_classes=dl.num_classes, lr=0.5, epochs=100, reg_lambda=1e-4)

        val_accs = []
        for i, (spikes, labels) in enumerate(dl.iter_split("val")):
            counts1, _ = simulate_get_counts(model, spikes, dt=args.dt)
            preds = predict_softmax(counts1.astype(np.float64), W, b)
            val_accs.append(accuracy(preds, labels))
            if i >= 10:
                break
        if val_accs:
            print(f"Validation accuracy (subset): {np.mean(val_accs):.3f}")
        else:
            print("No validation data available or failed to load.")
    else:
        # STDP on input->hidden weights using unsupervised batches, then train readout
        stdp_params = STDPParameters()
        stdp_state = initialize_stdp_state(batch_size=args.batch_size, num_pre=F, num_post=args.hidden)
        batches_done = 0
        for spikes, labels in dl.iter_split("train"):
            # Run layer 1 to get post spikes using current weights
            # We only need layer 1, so simulate up to hidden
            # Reuse simulate_get_counts to produce s1 via run_feedforward_two_layer but ignore layer2 by a trick
            # Instead, manually step first layer inside a small loop to capture spikes per timestep
            T, B, _ = spikes.shape
            # Lightweight one-layer sim: synapse + lif dynamics via engine functions
            from src.engine.sim import DenseLayerState, dense_layer_step

            state1 = DenseLayerState(
                syn_state=None,  # type: ignore
                lif_state=None,  # type: ignore
            )
            # Properly initialize states
            from src.models.synapse import initialize_exponential_state
            from src.models.lif import initialize_lif_state

            state1 = DenseLayerState(
                syn_state=initialize_exponential_state(B, args.hidden),
                lif_state=initialize_lif_state(B, args.hidden, model.layer1.lif_params),
            )

            for t in range(T):
                s_in = spikes[t]
                state1, s1 = dense_layer_step(s_in, state1, model.layer1, dt=args.dt)
                # STDP update uses pre/post spikes of this timestep
                stdp_state, new_w = stdp_step(s_in, s1, model.layer1.weights, stdp_state, stdp_params, dt=args.dt)
                model.layer1.weights = new_w

            batches_done += 1
            if batches_done >= args.max_batches:
                break

        # After STDP, train readout from hidden counts
        train_features: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        for spikes, labels in dl.iter_split("train"):
            counts1, _ = simulate_get_counts(model, spikes, dt=args.dt)
            train_features.append(counts1.astype(np.float64))
            train_labels.append(labels)
            if len(train_features) >= args.max_batches:
                break
        X_train = np.concatenate(train_features, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        W, b = train_softmax(X_train, y_train, num_classes=dl.num_classes, lr=0.5, epochs=100, reg_lambda=1e-4)

        val_accs = []
        for i, (spikes, labels) in enumerate(dl.iter_split("val")):
            counts1, _ = simulate_get_counts(model, spikes, dt=args.dt)
            preds = predict_softmax(counts1.astype(np.float64), W, b)
            val_accs.append(accuracy(preds, labels))
            if i >= 10:
                break
        if val_accs:
            print(f"Validation accuracy after STDP (subset): {np.mean(val_accs):.3f}")
        else:
            print("No validation data available or failed to load.")


if __name__ == "__main__":
    main()


