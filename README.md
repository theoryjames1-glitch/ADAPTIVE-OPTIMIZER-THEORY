
# 🌀 Adaptive Optimizer Theory

> **A universal wrapper for online adaptation of optimizer hyperparameters — without meta-loops, without schedulers, and completely outside the autograd graph.**

---

## 📊 Architecture Diagram

```mermaid
flowchart LR
    subgraph TrainLoop[Training Loop]
        L[Compute Loss] --> B[Backward Pass (gradients)]
        B --> SL[step_loss(loss)]
        SL --> OStep[optimizer.step()]
        OStep --> ZG[optimizer.zero_grad()]
    end

    subgraph AdaptiveLayer[Adaptive Layer (outside autograd)]
        SL -->|updates| Knobs
        Knobs -->|set| Groups[param_groups]
    end

    subgraph Optimizer[Base Optimizer (AdamW / SGD / RMSprop / Adagrad / 8-bit)]
        Groups --> OStep
    end

    style AdaptiveLayer fill:#f9f,stroke:#333,stroke-width:1px
    style Optimizer fill:#bbf,stroke:#333,stroke-width:1px
    style TrainLoop fill:#bfb,stroke:#333,stroke-width:1px
```
## 🚀 Motivation

Deep learning training relies heavily on optimizers such as **AdamW**, **SGD**, **RMSprop**, or **Adagrad**. These optimizers have fixed hyperparameters (learning rate, momentum, betas, alpha, weight decay) that are usually **hand-tuned** or controlled by schedulers.

The **Adaptive Optimizer Theory** proposes a clean alternative:

* **Keep the optimizer simple** (SGD, AdamW, RMSprop, Adagrad — or their 8-bit quantized versions from [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)).
* Add a **second step**, called `step_loss(loss)`, that adjusts hyperparameters dynamically based on the observed **loss signal**, completely **outside the computation graph**.

This means:

* No meta-training loops.
* No second-order gradients.
* No entangled graphs.
* Just **online, adaptive hyperparameters** that evolve as training progresses.

---

## ⚙️ Theory

1. **Optimizer** = applies parameter updates (AdamW, SGD, etc.).
2. **Adaptive Layer** = modifies optimizer hyperparameters before each step, based on rules applied to the current loss.
3. **Contract**:

```python
loss.backward()
adaptive.step_loss(loss)   # adapt knobs (lr, momentum, betas, etc.)
optimizer.step()           # apply update
optimizer.zero_grad()
```

That’s it — a clean separation of concerns.

---

## 🔧 Features

* Works with **standard PyTorch** optimizers (`AdamW`, `SGD`, `RMSprop`, `Adagrad`) and **bitsandbytes 8-bit** variants (`AdamW8bit`, `SGD8bit`, `RMSprop8bit`, `Adagrad8bit`, `PagedAdamW8bit`).
* Hyperparameters that can adapt:

  * `lr` (learning rate)
  * `momentum` (SGD, RMSprop)
  * `alpha` (RMSprop)
  * `betas` (AdamW)
  * `weight_decay` (AdamW, SGD, RMSprop, Adagrad)
* **Rule-based adaptation**:

  * `trend` → increase if loss improves, decrease otherwise
  * `variance` → scale down if loss variance is high
  * `inverse` → momentum increases as loss decreases
  * `relative` → boost momentum if improving, decay if not
  * `stable` → stabilize RMSprop alpha
  * `cosine` / `cosine_restart` → cosine annealing cycles
* Rules are **composable**: `"trend+variance"`, `"relative+cosine"`, etc.
* 100% **outside autograd** (`@torch.no_grad()`).

---

## 📦 Installation

```bash
pip install torch bitsandbytes
```

Copy `adaptive_opt.py` into your project.

---

## 🖥️ Usage

```python
import bitsandbytes as bnb
from torch.optim import AdamW, SGD, RMSprop, Adagrad
from adaptive_opt import AdaptiveOpt

# 1) build any optimizer as usual
optimizer = bnb.optim.AdamW8bit(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
)

# 2) wrap it with AdaptiveOpt
adaptive = AdaptiveOpt(
    optimizer,
    lr_rule="trend+cosine_restart",      # LR adapts with trend + cosine restarts
    momentum_rule="relative",            # Momentum adapts if available
    betas_rule=("relative","variance"),  # AdamW betas adapt separately
    wd_rule="trend",                     # Weight decay adapts
    cycle_length=500, cycle_mult=2
)

# 3) training loop
for step, (x, y) in enumerate(loader, 1):
    loss = model(x, y)
    loss.backward()

    adaptive.step_loss(loss)   # adapt knobs based on loss
    optimizer.step()           # apply update
    optimizer.zero_grad(set_to_none=True)

    if step % 50 == 0:
        print(f"Step {step} | Loss {loss.item():.4f} | LR {adaptive.lr:.6f}")
```

---

## 📊 Example Rules

* **SGD (8-bit)**:

  ```python
  adaptive = AdaptiveOpt(optimizer, lr_rule="trend", momentum_rule="relative")
  ```

* **RMSprop**:

  ```python
  adaptive = AdaptiveOpt(optimizer, lr_rule="variance", alpha_rule="stable")
  ```

* **AdamW**:

  ```python
  adaptive = AdaptiveOpt(optimizer,
                         lr_rule="trend+cosine",
                         betas_rule=("relative", "variance"),
                         wd_rule="trend")
  ```

---

## 🧪 Why it works

Instead of thinking of hyperparameters as constants, we treat them as **online adaptive parameters** that **react to training signals** in real time. They are:

* **Global** (shared across parameter groups).
* **Outside autograd** (no second-order cost).
* **Flexible** (rules are composable).

This blends the best of **schedulers** and **meta-learning** into a simple, universal adapter.

---

## 📖 Roadmap

* [ ] More adaptation rules (entropy-based, gradient-norm adaptive).
* [ ] Per-parameter group adaptation (not just global).
* [ ] Logging hooks for LR/momentum evolution.
* [ ] Visualization scripts (loss vs adaptive hyperparams).

---

## 📜 License

MIT — free to use, modify, and experiment with.

---

👉 Do you want me to also include a **diagram** (boxes for Optimizer ↔ Adaptive Layer ↔ Training Loop) in the README so it’s visually clear how the pieces interact?
