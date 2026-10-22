"""H62: the live training frontier, the durable free-GPU acquisition and the wider program.

H61 left a verified CPU data pipeline, a source-verified curriculum and a
measured negative result, but no trained checkpoint, because no free GPU was
ever obtained. H62 keeps the verified pipeline and its evidence intact and
adds the three things that were missing:

* `h62.colab_t4_watcher` is the durable, bounded, single-request allocation
  mechanism that asks the provider for a free T4 and records what the
  provider actually answers, so the frontier is a live mechanism instead of a
  configuration file.
* `h62.remote_h62_launch` is the pinned-source remote entry point that runs
  the H61 diagnostic against an exact commit and fails closed when the
  checkout or the training child fails.
* the capability, data, training-science and qualification work that turns
  the narrow drag-only deficit into a broader computer-use programme.

Nothing in this package claims that a GPU was obtained, that a checkpoint
exists, or that the model improved. Those are measured facts, and they are
recorded only when they are observed.
"""
