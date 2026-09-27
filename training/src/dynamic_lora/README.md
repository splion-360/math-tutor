# Dynamic LoRA

## The idea in plain language

A single tutor may need several ways of teaching. One style may be best for geometric intuition, another for symbolic proofs, and another for dense numerical work.

We could name those styles before training. That is what our first foundational, intermediate, and advanced adapters do.

Dynamic LoRA asks a different question: can useful specialists emerge from the training signals themselves, even when the reason is not difficulty?

The disagreement might instead come from notation, visual structure, proof style, topic, or some pattern we did not anticipate.

## What a LoRA adapter changes

A language model contains large weight matrices. For one matrix $W$, a normal layer computes:

$$
h' = Wh
$$

Fine-tuning every value in $W$ is expensive. LoRA freezes $W$ and learns a much smaller update:

$$
\Delta W = BA
$$

Here, $A$ and $B$ are thin matrices with rank $r$, where $r$ is much smaller than the size of $W$. The adapted layer becomes:

$$
h' = Wh + \frac{\alpha}{r}BAh
$$

The base model keeps its general knowledge. Each adapter acts like a small notebook containing a particular set of learned adjustments.

## Step 1: begin with one shared adapter

We first train one LoRA adapter across the whole Manim dataset. This gives every example the same starting point and avoids assuming that difficulty is the correct way to divide the work.

For a mini-batch $B_t$, training produces a loss $L_{B_t}$. The gradient for the LoRA parameters in layer $ℓ$ is:

$$
g_{ℓ,t} = \nabla_{\theta_ℓ} L_{B_t}
$$

The gradient points in the direction that would reduce the loss for that batch.

## Step 2: find the layers that matter most

Watching every model layer would add cost and noise. We first measure the gradient energy of each LoRA-bearing layer:

$$
E_ℓ = \lVert g_ℓ \rVert_2^2
$$

A high value means that the batch is asking for a larger change in that layer. We rank the layers by $E_ℓ$ and observe only the top $k$:

$$
S = \operatorname{TopK}(E_1, E_2, \ldots, E_L)
$$

This does not prove that a layer should split. It only tells us where the strongest training signals are appearing.

## Step 3: record the direction of each update

Raw gradients are too large to store at every training step. We compress each selected gradient into a smaller signature:

$$
z_{ℓ,t} = P_ℓ g_{ℓ,t}, \qquad z_{ℓ,t} \in \mathbb{R}^{m}
$$

The projection $P_ℓ$ maps a large gradient into $m$ values. Our current implementation uses a deterministic signed-bucket projection so the same run can be reproduced.

The signature is an observation tool. It gives us a compact view of direction, but it is not treated as an exact copy of the original gradient.

## Step 4: measure agreement and conflict

We compare two signatures using cosine similarity:

$$
c_{ℓ,t} =
\frac{z_{ℓ,t}^\top z_{ℓ,t-1}}
{\lVert z_{ℓ,t} \rVert_2\,\lVert z_{ℓ,t-1} \rVert_2}
$$

The result is easy to interpret:

- $c \approx 1$: the updates point in a similar direction.
- $c \approx 0$: the updates are mostly unrelated.
- $c < 0$: improving one batch may work against the previous batch.

The code currently compares consecutive optimizer steps and records the signatures, cosine values, and conflict rate. These artifacts can later be plotted or studied in two dimensions.

One negative cosine is not enough to create a specialist. Mini-batches are noisy, so a real split decision must depend on a pattern that persists.

## Step 5: smooth the evidence

The next stage will maintain a smoothed direction for each candidate group:

$$
\mu_{j,t} = \beta\mu_{j,t-1} + (1-\beta)z_t
$$

The value $\beta$ controls memory. A larger value changes the centroid slowly; a smaller value reacts more quickly to recent batches.

We can measure the separation between two candidate centroids with:

$$
d(\mu_1, \mu_2) = 1 - \cos(\mu_1, \mu_2)
$$

A split should only be considered when the separation stays above a chosen threshold for several steps and both groups have enough examples.

## Step 6: make one controlled split

The first experiment uses a binary split: one parent adapter may create two children. This keeps the decision easy to inspect and avoids an open-ended collection of specialists.

If the parent has a fixed adapter budget $R$, the two children must share it:

$$
r_1 + r_2 = R
$$

The split is accepted only if the child pair improves held-out performance enough to justify the added routing complexity.

## Step 7: learn the inference router

During training, the split process tells us which child adapter handled each example. These assignments act as oracle labels:

$$
a_i \in \{1, 2\}
$$

A lightweight router then learns to predict the adapter from the prompt alone:

$$
p(a \mid x) = \operatorname{Router}(x)
$$

At inference time, the gradient is unavailable because there is no correct answer or loss yet. The router therefore chooses:

$$
a^* = \arg\max_a p(a \mid x)
$$

This separates discovery from serving: gradients discover the specialists during training, while the prompt router selects them for a learner's new question.

## What exists today

The repository currently includes:

- shared LoRA training for Qwen3-4B;
- gradient-energy ranking and top-$k$ layer selection;
- projected gradient signatures during optimizer steps;
- cosine and conflict metrics in W&B;
- raw JSONL signatures and a summary manifest;
- a Modal smoke-training path.

The EMA split gate, binary child training, oracle assignment export, learned router, and dynamic serving path are the next stages. They should not be presented as completed experiments yet.

## What we want to visualize

The saved experiment data can support:

- training and held-out loss curves;
- gradient energy by layer;
- cosine similarity and conflict rate over time;
- a two-dimensional view of gradient signatures using PCA, UMAP, or t-SNE;
- the prompts associated with each discovered cluster;
- parent-versus-child adapter performance after a split.

Cluster names should be assigned only after inspecting the examples. If a group appears to represent geometry, symbolic density, or difficulty, that is a post-hoc interpretation rather than a label supplied to the splitting rule.

## An important experiment detail

The training records retain difficulty as metadata for splitting and analysis, but the model prompt contains only the topic and task. The predefined difficulty proxy is not supplied as an input feature when probing or training the model.

For commands, configuration, data rules, and artifact locations, see the [training guide](../../README.md).
