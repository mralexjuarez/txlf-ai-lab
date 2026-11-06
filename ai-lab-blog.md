# Your Own Private Stack: Local LLMs and RAG with Lima and Ollama

*This is the companion post for my two-hour hands-on tutorial. Everything below — the Lima template, the sample documents, and all four lab scripts — lives in the workshop repo, and every line of it runs on a laptop with no cloud account, no API key, and no GPU. If you're attending the workshop, please complete [Lab 0](#lab-0-pre-work-do-this-before-the-workshop) beforehand; the downloads total about 3 GB and conference wifi will not be kind to us.*

There's a version of "getting into AI" that starts with a credit card: sign up for an API, paste a key into an environment variable, and send your data to someone else's computer. This tutorial is the other version. By the end of it you will have a small, self-contained lab — a Linux virtual machine defined by one YAML file — that runs a language model locally, indexes your own documents, and answers questions about them. When we're done, you can delete the whole thing with one command and rebuild it with another. That's the point.

We'll assemble four pieces:

1. **Lima** gives us a repeatable, disposable Linux environment — the same one for everybody, whether your laptop runs macOS or Linux.
2. **Ollama** runs a language model inside that environment and exposes it as a plain HTTP API.
3. **Chroma**, a small embedded vector database, stores our documents as embeddings.
4. **About 150 lines of plain Python** connect retrieval to generation. No frameworks. You will see every moving part.

Here's the architecture we're building toward:

```
your laptop (macOS or Linux)
│
│  limactl shell / curl localhost:11434        ~/ai-lab (shared folder)
▼                                              ▲
┌─ Lima VM ─ Ubuntu 24.04 LTS ─────────────────┼──────────────┐
│                                              │              │
│   Ollama (systemd service, port 11434)       │              │
│   ├── llama3.2:3b        ← generates answers │              │
│   └── nomic-embed-text   ← turns text into vectors          │
│                                                             │
│   Chroma (embedded, ./chroma_db)  ← stores document vectors │
│                                                             │
│   ask.py:  question → embed → retrieve → prompt → answer    │
└─────────────────────────────────────────────────────────────┘
```

And here's how the two hours break down:

| Time | Segment |
|---|---|
| 0:00–0:10 | Why local? The four moving parts |
| 0:10–0:35 | Lima — primer + **Lab 1** (kick the tires on a VM) |
| 0:35–0:55 | Ollama — primer + **Lab 2** (talk to a model, CLI and API) |
| 0:55–1:15 | RAG — primer + **Labs 3a–3c** (embeddings, ingest, search) |
| 1:15–1:50 | Bringing it together — **Lab 4** (ask your documents) |
| 1:50–2:00 | Where it breaks, where to go next, Q&A |

---

## Lab 0: Pre-work (do this before the workshop)

You need: macOS 13.5+ (Intel or Apple Silicon) or Linux with KVM, 16 GB of RAM recommended (8 GB works — see the note below), and about 15 GB of free disk.

**1. Install Lima.** On macOS, or on Linux with Homebrew:

```bash
brew install lima
```

On Linux without Homebrew, grab the tarball from the [Lima releases page](https://github.com/lima-vm/lima/releases) and extract it under `/usr/local`. Linux hosts also need QEMU (`sudo apt install qemu-system-x86` or your distro's equivalent) and your user in the `kvm` group. Verify with `limactl --version` — anything v1.0+ works, v2.x is current.

**2. Get the workshop folder in place.** Unzip or clone the repo so the files live at `~/ai-lab`. That path matters twice over: the template shares exactly that host folder, and pins it to `~/ai-lab` *inside* the VM too. Clone somewhere else and Lima cheerfully creates an empty `~/ai-lab` on the host, mounts that, and boots a VM with none of the workshop files in it — which is what the readiness probe in the current template exists to catch.

**3. Create the VM.**

```bash
cd ~/ai-lab
limactl start ./ai-lab.yaml
```

First boot downloads an Ubuntu cloud image (~600 MB) and installs Ollama inside the VM automatically. If your machine has only 8 GB of RAM, edit `ai-lab.yaml` first and change `memory: "8GiB"` to `"4GiB"` — the 3B model still fits.

**4. Pull the two models (~2.3 GB total).**

```bash
limactl shell ai-lab ollama pull llama3.2:3b
limactl shell ai-lab ollama pull nomic-embed-text
```

**5. Set up Python inside the VM.**

```bash
limactl shell ai-lab            # from the host, enter the VM
cd ~/ai-lab                     # the shared folder, in the VM
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install requests chromadb
```

The venv lives in the VM's home directory rather than the shared folder, because shared-folder filesystems and virtualenvs don't always get along.

You may see `cd: /Users/you/ai-lab: No such file or directory` as the shell opens. That is Lima trying your host path, which does not exist in the guest because the folder is mounted at `~/ai-lab`; the shell falls back to the VM's home and everything still works. `cd ~/ai-lab` is always the way in.

**6. Smoke test.** Still inside the VM with the venv active:

```bash
cd ~/ai-lab
python3 smoke_test.py
```

Three `[ok]` lines means you're ready. If something fails, the [troubleshooting section](#troubleshooting) at the end covers the usual suspects.

---

## Part 1: Lima — the environment as a file

### What Lima is

Lima ("**Li**nux **Ma**chines") is a command-line tool that launches Linux virtual machines with two conveniences that normally require fiddly setup: automatic file sharing between host and guest, and automatic port forwarding. If you've used WSL2 on Windows, the experience is similar — except Lima runs on macOS and Linux, and the entire machine definition is a YAML file you can check into git.

That last property is why we're using it. A README that says "install Python 3.12, then Ollama, then..." rots. A Lima template *is* the environment. Everyone in this room starts the same VM from the same file and gets the same result, regardless of what's installed — or broken — on their host.

### A short history

Lima was started in **May 2021 by Akihiro Suda**, a containerd maintainer, with a modest initial goal: give macOS users a good way to run containerd and its companion CLI, nerdctl, since containers are a Linux kernel feature and Macs don't have one. The tool turned out to be far more generally useful. It joined the **CNCF as a sandbox project in September 2022**, and other tools quietly built on top of it — Colima, Rancher Desktop, and AWS's Finch all use Lima under the hood, so you may have been running it for years without knowing.

Two recent milestones matter for us. Lima **v1.0 shipped in November 2024**, switching the default macOS driver to Apple's Virtualization.framework for much faster file sharing. Then Lima was promoted to a **CNCF incubating project in October 2025**, and **v2.0 landed that November with an explicit new focus: AI workflows**. The project's own headline use case for v2 is running AI tools inside a VM so that a misbehaving model, script, or agent can't touch your host — if it runs `curl | sh` on something malicious, the blast radius is a VM you can delete. There's even a new `krunkit` driver that gives local LLMs GPU acceleration on Apple Silicon. In other words: using Lima as an AI sandbox isn't a clever trick we invented for this workshop. It's what the project itself now says it's for.

### Why a VM for AI work, specifically

Three reasons, in increasing order of importance:

**Isolation.** The standard install command for half the AI ecosystem is `curl -fsSL something.sh | sh`. Inside a VM, that's a shrug instead of a risk assessment.

**Repeatability.** Model runtimes, Python packages, and system libraries move fast and break each other. Pinning the whole OS in a template ends the "works on my machine" conversation.

**Disposability.** The most underrated feature. When your environment costs one command to rebuild, you experiment more freely. Break it on purpose. You'll see.

### The `limactl` CLI in five commands

```bash
limactl start <template>   # create + boot a VM (from a name, file, or URL)
limactl list               # what exists, what's running
limactl shell <name>       # open a shell inside a VM
limactl stop <name>        # shut down (state preserved)
limactl delete <name>      # gone. completely.
```

One behavior worth knowing: `limactl shell` tries to open the guest shell in your *current host directory*, provided that directory is mounted into the VM. Our workshop folder is mounted at `~/ai-lab` in the guest rather than at its host path, so current Lima releases print a harmless `cd:` error and land you in the VM's home instead; `cd ~/ai-lab` gets you to the files. (Lima master maps the host path through the mount point, which makes that hop seamless.)

### Lab 1: Kick the tires

We'll create a throwaway VM from Lima's built-in default template, poke at it, and destroy it. Everything here is safe — your real lab VM (`ai-lab`) isn't involved.

```bash
# 1. Create a scratch VM from the built-in default template
limactl start --name=scratch template:default
#    (choose "Proceed with the current configuration" at the prompt)

# 2. Look around
limactl list
limactl shell scratch
```

You're now inside a Linux VM. Try:

```bash
uname -a                    # a real Linux kernel
whoami                      # your host username, mirrored into the guest
ls ~                        # hmm — where are your files?
```

Your host home directory is mounted into the guest — but try to write to it:

```bash
touch ~/some-host-file.txt
# touch: cannot touch '…': Read-only file system
```

**This is Lima's most important default:** your home directory is shared *read-only*. The guest can read your files but cannot modify them. If you want writable access, you must ask for it explicitly, per directory, in the template — which is exactly what our `ai-lab.yaml` does for one folder only. Sensible sandbox defaults, out of the box.

Now practice the disposability muscle:

```bash
exit                        # back to the host
limactl stop scratch
limactl delete scratch
limactl list                # scratch is gone; ai-lab remains
```

Note how fast the second VM was to create: Lima caches downloaded images, so new VMs from the same base cost seconds, not another 600 MB.

### Anatomy of our template

Here is the entire definition of the lab you're sitting in — `ai-lab.yaml`:

```yaml
# ai-lab.yaml — a repeatable local Linux lab for LLMs + RAG
minimumLimaVersion: "1.0.0"

# Pin the OS for repeatability. Everyone gets the same Ubuntu LTS,
# on Intel or Apple Silicon, today or a year from now.
images:
  - location: "https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-amd64.img"
    arch: "x86_64"
  - location: "https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-arm64.img"
    arch: "aarch64"

# Right-sized for a 3B model. Low on host RAM? Drop memory to "4GiB".
cpus: 4
memory: "8GiB"
disk: "30GiB"

# Share ONLY the workshop folder, and make it writable.
# `location` is the folder on the host; `mountPoint` is where it shows up in the
# guest. Lima defaults mountPoint to the host path (e.g. /Users/you/ai-lab), so
# we set it explicitly and the files appear at ~/ai-lab inside the VM.
# (In mountPoint, {{.Home}} is the guest's home, not the host's.)
mounts:
  - location: "~/ai-lab"
    mountPoint: "{{.Home}}/ai-lab"
    writable: true

# Make the Ollama API inside the VM reachable at localhost:11434 on the host.
portForwards:
  - guestPort: 11434
    hostPort: 11434

# Everything the lab needs, installed automatically on first boot.
provision:
  - mode: system
    script: |
      #!/bin/bash
      set -eux
      export DEBIAN_FRONTEND=noninteractive
      apt-get update
      apt-get install -y python3-pip python3-venv
      curl -fsSL https://ollama.com/install.sh | sh
      # The installer skips starting the service while the VM is still booting
      # ("systemd is not running"), so start it explicitly.
      systemctl daemon-reload
      systemctl enable --now ollama

# Don't report "ready" until Ollama is answering and the share is populated.
probes:
  - description: "ollama API to be ready"
    script: |
      #!/bin/bash
      until curl -sf http://localhost:11434/api/version >/dev/null 2>&1; do
        sleep 3
      done
  # An empty shared folder is otherwise silent: the VM boots fine and you only
  # find out in step 6, when smoke_test.py is missing.
  - description: "workshop files to be visible at ~/ai-lab"
    script: |
      #!/bin/bash
      test -f "{{.Home}}/ai-lab/smoke_test.py"
    hint: |
      The VM is up, but ~/ai-lab/smoke_test.py is not there, which means the
      folder Lima shared is empty or in the wrong place.
      Check that ai-lab.yaml sits in the same folder as smoke_test.py (see step 2
      of README.md), then recreate the instance:
        limactl delete ai-lab
        limactl start ./ai-lab.yaml
      The VM is still usable meanwhile: `limactl shell ai-lab`.
```

Read it top to bottom and notice what each block buys us. The **pinned image** means the workshop behaves the same in a year (the built-in `template:default` tracks the newest Ubuntu, which is great for daily use and wrong for a reproducible lab). The **mount** grants writable access to one folder instead of your whole home directory, and decides where that folder lands in the guest. Without `mountPoint`, Lima mounts a directory at its host path: `/Users/you/ai-lab` on macOS, `/home/you/ai-lab` on Linux. Neither is the guest's own home — Lima gives the guest user a home of `/home/you.guest` — so `~` inside the VM would not be where your files are. `{{.Home}}` in `mountPoint` is the *guest's* home (Lima evaluates `location` on the host and `mountPoint` in the guest), so the workshop folder is at `~/ai-lab` either way. The **port forward** is how your host tools will reach the model. The **provision script** is where that scary `curl | sh` goes to live safely. And the **probes** make `limactl start` block until Ollama is genuinely up *and* the shared folder actually contains the workshop files, so "it started" means "it works."

That file is the deliverable of this workshop as much as any script is. When you get home, everything you build on top of it travels with it.

---

## Part 2: Ollama — a runtime, a registry, and an API

### What Ollama is

Ollama is the piece that actually runs the language model. It's three things in one small binary: a **runtime** that executes models efficiently on ordinary CPUs and GPUs, a **registry client** with a Docker-like pull/push workflow against a large public model library, and an **HTTP API server** that everything else on your machine can talk to. On Linux — including inside our VM — it installs as a systemd service and listens on port 11434.

### A short history

Two projects made this moment possible. In early 2023, Meta released the LLaMA models to researchers, and **Georgi Gerganov's llama.cpp** promptly demonstrated something the industry had assumed away: with aggressive quantization, capable language models could run on consumer CPUs — laptops, even phones. A huge tinkerer ecosystem formed around it, along with the **GGUF** file format for packaged, quantized model weights.

But llama.cpp was (gloriously) a hacker's tool: compile flags, model conversion scripts, checksum spelunking. In mid-2023, **Jeffrey Morgan and Michael Chiang launched Ollama** to close that gap. Their pedigree tells you the design philosophy: they previously built Kitematic, the friendly Docker GUI that Docker acquired. Ollama gave local models the Docker experience — `ollama pull`, `ollama run`, a curated library at ollama.com, and a `Modelfile` format for customizing models the way a Dockerfile customizes images. It became the default on-ramp for local LLMs, and its API (including an OpenAI-compatible endpoint) became a de facto standard that dozens of tools integrate with. Originally a wrapper around llama.cpp, it has since grown engine work of its own, but the promise is unchanged: one command between you and a running model.

### Quantization in one paragraph

A model's "3B" means three billion parameters — learned numbers. Stored at full 16-bit precision, that's roughly 6 GB before you've generated a single word. Quantization stores each parameter in ~4 bits instead, shrinking our model to about 2 GB with a modest quality cost. That's the entire trick that makes "LLM on a laptop, no GPU" a sentence you can say. When you `ollama pull llama3.2:3b`, you're getting a 4-bit quantized GGUF by default.

### Lab 2: Talk to the model

Get inside the lab VM:

```bash
limactl shell ai-lab
cd ~/ai-lab
```

**First, the CLI.** See what's installed and start a conversation:

```bash
ollama list                 # the two models you pulled in Lab 0
ollama run llama3.2:3b
```

You're now chatting with a language model running entirely inside a VM on your laptop. Ask it something. Notice the speed — this is CPU inference, and it's *fine*. Type `/bye` to exit. Two more useful commands: `ollama ps` shows loaded models and memory use; models unload automatically after a few idle minutes.

**Second, the API.** The CLI is a client of the same HTTP server everything else uses. Still inside the VM:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Reply with exactly five words about Linux.",
  "stream": false
}'
```

The endpoints we'll use all week: `/api/generate` (one-shot completion), `/api/chat` (message-based, with roles), `/api/embed` (text to vectors — the star of Part 3), and `/api/tags` (list models). There's also an OpenAI-compatible `/v1` surface, which means a large fraction of existing AI tooling can point at your VM instead of a cloud.

**Now the magic moment.** Exit the VM (`exit`) and run the *same curl from your host terminal*:

```bash
curl http://localhost:11434/api/version
```

It answers. That's the `portForwards` block from our template doing its job: the model is isolated inside the VM, but its API is a first-class citizen of your laptop. Any editor plugin, script, or app on your host that speaks HTTP can now use a local model — and the model still can't touch your filesystem.

---

## Part 3: RAG — giving the model an open book

### The problem

Let's demonstrate the problem before naming it. Our sample corpus for today is the internal wiki of a small fictional candle company, Kiln & Wick Candle Co. — four short markdown files in `docs/` covering the employee handbook, product catalog, fragrance formulas, and an operations runbook. (Swap in your own docs later; the pipeline doesn't care.)

Ask the raw model a question about it:

```bash
ollama run llama3.2:3b "What is Kiln & Wick Candle Co.'s return policy for a candle that has been lit?"
```

You'll get one of two failure modes: an honest "I don't have information about that company," or — worse — a confident, plausible, entirely invented returns policy. Neither should surprise us. A language model's knowledge is baked into its parameters at training time. It has read a snapshot of the public internet; it has never read *your* documents, and it stopped learning at its training cutoff.

Three fixes suggest themselves. Retraining or fine-tuning the model on your documents is expensive, slow, and has to be redone every time a document changes. Pasting all your documents into every prompt works right up until your corpus exceeds the model's context window — and even before that, you pay in speed and in the model's attention getting diluted. The third option is the one that stuck: keep the documents *outside* the model, and at question time, **retrieve only the relevant passages and place them in the prompt**. Retrieval-Augmented Generation. The model takes an open-book exam, and we hand it the right pages.

### A short history

The name comes from a specific paper: **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Lewis et al., 2020**, out of Facebook AI Research. The idea has older roots — open-domain question-answering systems like DrQA (2017) paired search with reading comprehension, and 2020 also produced REALM and Dense Passage Retrieval, which established that *neural embeddings* could out-retrieve classic keyword search. But RAG stayed a research topic until ChatGPT made the limitations of parametric knowledge everyone's problem. Through 2023, "chat with your PDFs" became the industry's hello-world, a whole crop of vector databases appeared to serve it, and RAG settled in as the standard architecture for grounding models in private or fresh data. The field has since piled on refinements — hybrid keyword-plus-vector search, rerankers, agentic retrieval loops — but the core loop you'll build today is the same one in the 2020 paper, and it's still what those refinements refine.

### Embeddings: meaning as coordinates

The load-bearing concept in RAG is the **embedding**: a model that reads text and outputs a vector — a long list of numbers — such that *texts with similar meaning get vectors that point in similar directions*. Not similar spelling. Similar *meaning*. "The candle smells like lavender and cedar" and "This wax melt has notes of lavender and wood" share almost no words, but their vectors sit close together, and both sit far from "Restart the server after updating the config file."

We measure "close" with cosine similarity: 1.0 means pointing the same way, near 0 means unrelated. Our embedding model is `nomic-embed-text`, a 274 MB open model that produces 768-dimensional vectors and consistently benchmarks above the paid embedding APIs of a couple of years ago.

### Lab 3a: See an embedding

Inside the VM, venv active:

```bash
python3 embed_demo.py
```

The script is short enough to read in full:

```python
import requests

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    return r.json()["embeddings"]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    return dot / (mag_a * mag_b)

sentences = [
    "The candle smells like lavender and cedar.",
    "This wax melt has notes of lavender and wood.",
    "Restart the server after updating the config file.",
]
vectors = embed(sentences)
# ...prints vector length and pairwise similarities
```

Expect the two candle sentences to score dramatically higher with each other than either does with the sysadmin sentence. That gap — *meaning made measurable* — is the entire foundation of what follows. Note also that `cosine()` is eight lines of arithmetic. No magic anywhere.

### Chunking, and why we need a database

Two practical questions stand between us and a working system. First: what do we embed? Whole files are bad retrieval units — a file about "operations" matches everything vaguely operational. We split documents into **chunks** of a few hundred characters, so each vector represents roughly one topic and retrieval can be precise. Our chunker is deliberately naive — split on blank lines, merge paragraphs up to 400 characters — and that's a feature: you can read it, and it works. (Real systems fight harder here; chunk boundaries are where RAG bugs live. More on that at the end.)

Second: where do the vectors go? For 17 chunks you could compare against every vector in a Python loop. For a million chunks you want a **vector database**: persistence plus data-structures for fast nearest-neighbor search, typically with metadata filtering. We're using **Chroma**, an open-source vector database released in early 2023, in its embedded mode — it's a pip install that persists to a local folder, backed by SQLite. No server, no configuration, ideal for a lab. (Qdrant, pgvector, and friends await you when you outgrow it; the concepts transfer exactly.)

### Lab 3b: Ingest the documents

```bash
python3 ingest.py
# Embedding 17 chunks from 4 files...
# Done. Collection 'docs' in ./chroma_db now holds 17 chunks.
```

The core of `ingest.py`:

```python
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},  # smaller distance = more similar
)

ids, texts, metadatas = [], [], []
for path in sorted(pathlib.Path("docs").glob("*.md")):
    for i, piece in enumerate(chunk(path.read_text())):
        ids.append(f"{path.stem}-{i}")
        texts.append(piece)
        metadatas.append({"source": path.name})

collection.upsert(ids=ids, documents=texts,
                  embeddings=embed(texts), metadatas=metadatas)
```

Every chunk is stored three ways at once: its text (to put in a prompt later), its vector (to search by), and its metadata (to cite sources). The `upsert` means re-running after editing a doc just works. Take a moment to look at what appeared: `ls chroma_db/` — your knowledge base is a folder inside your project, and because that folder is our writable Lima mount, you can see it from the host too.

### Lab 3c: Search — retrieval with no LLM at all

This is the step people skip, and skipping it is why people can't debug their RAG systems. Retrieval works *without* a language model, so test it without one:

```bash
python3 search.py "what do we do when the wick supplier is out of stock?"
```

```
Q: what do we do when the wick supplier is out of stock?

--- ops-runbook.md  (distance 0.334)
    ## When the wick supplier is out of stock Our primary wick is the CD-16...

--- handbook.md  (distance 0.512)
    ## Wholesale accounts Wholesale pricing starts at 50 units per order...
```

The question was embedded with the *same model* as the chunks, Chroma returned the nearest neighbors, and the right passage came back first — with a visibly better (smaller) distance than the runners-up. Try questions phrased nothing like the source text ("my candle showed up as a puddle, what now?") and watch semantic search earn its keep. Try a question the docs can't answer and notice the distances get worse but *something still comes back* — retrieval always returns its best guess. Remember that; it's foreshadowing.

---

## Part 4: Bringing it together

### The whole loop

Everything so far snaps together into one pipeline:

```
question ─→ embed it ─→ nearest chunks ─→ build a prompt ─→ generate ─→ answer + sources
          (nomic-embed)   (Chroma)        (an f-string!)   (llama3.2)
```

The step that sounds most mysterious — "augmenting the generation" — is the least mysterious in practice. Here is the complete heart of `ask.py`:

```python
# 1. Embed the question.
r = requests.post(f"{OLLAMA}/api/embed",
                  json={"model": EMBED_MODEL, "input": question})
query_vector = r.json()["embeddings"][0]

# 2. Retrieve the most relevant chunks.
collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("docs")
results = collection.query(query_embeddings=[query_vector], n_results=TOP_K)
chunks = results["documents"][0]
sources = sorted({m["source"] for m in results["metadatas"][0]})

# 3. Build the prompt. This is the entire "framework".
context = "\n\n---\n\n".join(chunks)
messages = [
    {"role": "system", "content":
        "Answer the question using ONLY the provided context. "
        "If the context does not contain the answer, say you don't know."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
]

# 4. Generate.
r = requests.post(f"{OLLAMA}/api/chat",
                  json={"model": CHAT_MODEL, "messages": messages, "stream": False})
answer = r.json()["message"]["content"]
```

Look hard at step 3. The "augmentation" in Retrieval-Augmented Generation is string formatting. We paste the retrieved evidence into the prompt, above the question, with instructions to use only that evidence and to admit ignorance otherwise. Every RAG framework you will ever meet is, at its center, a fancier version of this f-string. Learning it this way first means the frameworks will never be magic to you — they'll be conveniences.

### Lab 4: Ask your documents

```bash
python3 ask.py "what is the return policy on a candle that has been lit?"
```

```
Q: what is the return policy on a candle that has been lit?

A: Once a candle has been lit it can't be returned, but candles with a
   manufacturing defect — such as tunneling, a clogged wick, or cracked
   glass — will be replaced within 90 days of purchase. Email
   care@kilnandwick.example with a photo of the defect.

Sources: handbook.md
```

Now put it side by side with the raw model from the start of Part 3 — same model, same question, night and day. The parameters didn't change; the *prompt* changed, because retrieval put the truth in front of the model. More to try:

```bash
python3 ask.py "which scent has cedarwood in its base notes?"
python3 ask.py "what temperature do we pour wax at?"
python3 ask.py "how should wax melts be packed for shipping?"
python3 ask.py "what is our policy on remote work?"   # not in the docs!
```

That last one matters most. Watch the system (usually!) say it doesn't know — because our system prompt told it to, and because we gave it nothing to bluff with. Then remember the foreshadowing from Lab 3c: retrieval *always returns something*, so if the top chunks are irrelevant enough, a small model can still get creative. Which brings us to:

### Where it breaks (an honest list)

Your pipeline has exactly three places to fail, and now you can debug each in isolation. **Retrieval misses:** the right chunk exists but doesn't come back — test with `search.py`, tune chunk sizes, consider hybrid keyword+vector search or a reranker. **Chunking splits facts:** the answer straddles a chunk boundary and neither half makes sense alone — smarter, structure-aware splitting helps, and this is where real systems invest. **Generation wobbles:** the evidence is right there and a 3B model still fumbles the synthesis — a bigger model (see below) usually fixes what prompt-tinkering can't. Evaluating all this systematically is a genuine engineering field now; search for "RAG evaluation" when you're ready.

---

## Take it home

You leave with a folder and a YAML file, which means you leave with everything. Some directions to grow it, roughly in order of effort:

**Swap in your own documents.** Drop markdown, text, or converted files into `docs/`, re-run `ingest.py`, and ask away. Meeting notes, a wiki export, your homelab documentation — this afternoon's pipeline handles it unchanged. Your documents never leave the VM.

**Swap the model.** The nice thing about `ollama pull` is that upgrading intelligence is one command. With 8 GB+ of VM memory try `qwen3:4b`, `gemma3:4b`, or `phi4-mini`; with more, the 7–8B tier is a real step up in answer quality. Change one constant in `ask.py`.

**Stream the answer.** Set `"stream": True` and read the response line by line — instant perceived speed-up.

**Give it a face.** Point Open WebUI (or any OpenAI-compatible client) at your forwarded port and you have a private ChatGPT-alike over your own docs.

**Go faster on Apple Silicon.** Lima v2's `krunkit` driver enables GPU acceleration for VMs on macOS — worth exploring once the basics feel comfortable.

**Share the lab.** The `ai-lab.yaml` file is the environment. Send it to a colleague, and one `limactl start` later they have your setup. That was the promise of the workshop's title, and now it's just a fact about your laptop.

And when the experiment goes sideways — and if you're doing it right, it eventually will — the fix is the whole thesis of the afternoon:

```bash
limactl delete ai-lab
limactl start ./ai-lab.yaml     # good as new, five minutes
```

---

## Troubleshooting

**`limactl start` fails or hangs on the readiness probe.** The provision script may still be downloading Ollama on slow wifi — give it a few minutes. To investigate, run `limactl shell ai-lab -- systemctl status ollama`, or check the logs under `~/.lima/ai-lab/` on the host. If the service shows `inactive` and `disabled`, Ollama installed but never started, because its installer skips that step while the VM is still booting (look for `systemd is not running` in the provisioning log). The YAML above handles this with `systemctl enable --now ollama`. To fix a VM built without that line, run `limactl shell ai-lab sudo systemctl enable --now ollama`.

**`limactl start` reports "workshop files to be visible at ~/ai-lab" as unmet, or `~/ai-lab` in the VM is empty.** The host folder Lima was told to share is empty or missing. Lima shares `~/ai-lab` from the host and quietly creates that directory if it does not exist, so a clone kept anywhere else produces a VM with no workshop files at all. Put the repo at `~/ai-lab` (a symlink to your checkout works fine), or point `location` in `ai-lab.yaml` at the real folder, then rebuild: `limactl delete ai-lab && limactl start ./ai-lab.yaml`.

**A shell opens with `cd: /Users/you/ai-lab: No such file or directory`.** Cosmetic. Lima tries to start the shell in your host directory, while the workshop folder is mounted at `~/ai-lab` in the guest, so that `cd` cannot succeed and the shell falls back to the VM's home. Run `cd ~/ai-lab`. A VM created from the older template has the mirror-image problem — the folder exists in the guest only at its host path (`/Users/you/ai-lab`, or `/home/you/ai-lab` on a Linux host) and `~` looks empty — and needs `limactl delete ai-lab && limactl start ./ai-lab.yaml` to pick up the new `mountPoint`.

**Port 11434 is already in use on the host.** You have Ollama installed natively on your host — the forward can't bind. Either stop the host service (`brew services stop ollama` on macOS, `sudo systemctl stop ollama` on Linux) or edit `hostPort` in the template to `11435` and use that from the host.

**`curl localhost:11434` works inside the VM but not from the host.** Check `limactl list` shows the VM `Running`, then `limactl stop ai-lab && limactl start ai-lab` to re-establish forwards. See also the port-conflict item above.

**Chroma throws SQLite/locking errors.** Some shared-folder filesystems (notably on Linux/QEMU hosts) upset SQLite. Move the database into the VM's native disk: `export CHROMA_PATH=~/chroma_db`, re-run `ingest.py`, and keep that variable set for `search.py`/`ask.py` — the scripts already honor it.

**Generation is very slow or the VM feels starved.** First token after idle includes model-load time (tens of seconds on CPU) — that's normal; subsequent replies are faster. If it's *all* slow, your VM may have too little memory: 4 GiB runs `llama3.2:3b` adequately, but close other apps, and don't run bigger models there.

**Linux: `limactl start` complains about virtualization.** Ensure KVM is available (`ls /dev/kvm`), QEMU is installed, and your user is in the `kvm` group (`sudo usermod -aG kvm $USER`, then log out and in).

**"I didn't do the pre-work" (workshop day).** Sit with a neighbor who did for the labs — the walkthrough is designed to be followable on one machine per pair — and run Lab 0 tonight; everything here works identically at home.

---

*Slides, the Lima template, sample docs, and all scripts: [workshop repo link]. Questions or corrections: [your contact]. Lima is a CNCF incubating project at [lima-vm.io](https://lima-vm.io); Ollama lives at [ollama.com](https://ollama.com); Chroma at [trychroma.com](https://www.trychroma.com); and the paper that named the pattern is Lewis et al., 2020, ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."](https://arxiv.org/abs/2005.11401)*
