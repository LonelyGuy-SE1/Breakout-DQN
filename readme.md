# Breakout DQN: First-Principles Pipeline

A (kind of) first principles pipeline build of a Reinforcement Learning agent for Atari Breakout, using PyTorch.

_(Note: The current iteration utilizes DQN to establish off policy mathematical foundations)._

The system is decoupled into isolated modules to separate the environment constraints from the calculus engine.

## Execution

### 1. Dependencies

Ensure you have the core tensor engines and the Atari emulators installed.

```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

### 2. Synchronization & Training (Cloud/GPU Recommended)

To initiate the learning loop, execute the orchestrator. For full convergence, this should be run on a dedicated tensor accelerator (e.g., Nvidia T4/P100) for ~5,000,000 frames.

```bash
python -m Orchestrator.train
```

_Outputs: dqn_brain.pth (The serialized neural weights)._

### 3. Deployment & Inference (Local CPU)

To watch the trained agent execute in reality with 100% exploitation (no random exploration):

```bash
python -m Tests.evaluate
```

The current implementation relies on a brute force spatial memory architecture (LazyReplayBuffer). Potential Future iterations of this logic will bypass raw pixel processing by dreaming in latent space, specifically exploring:

- Variational Autoencoders (VAEs) to compress spatial reality into denser mathematical representations.
- World Models to allow the agent to hallucinate future states and train within its own neural simulation prior to interacting with the live environment.
