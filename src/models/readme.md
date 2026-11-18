All detection models are placed in this folder.
Models include:
  1. MAP (ML)
  2. MMSE
  3. SIC (via QR decomposition)
  4. partial MAP [1]

[1] Deep Learning-Based Low Complexity MIMO Detection via Partial MAP

Configure each detector with `updatePossibleTXSymbols(symbols, bits)`.
For a shared constellation of size M, use a symbol vector of shape (M,)
and one row of B bit labels per symbol, in an (M, B) array.
For MMSE with a joint symbol dictionary of shape (N_tx, M), supply labels
of shape (N_tx, M, B) in the same candidate order.

Pass received sequences as (N_samples, N_rx). Detection returns integer
bits of shape (N_tx, N_samples, B).
