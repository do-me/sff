# SemanticFileFinder (sff)

[![crates.io](https://img.shields.io/crates/v/sff.svg)](https://crates.io/crates/sff)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)
<!-- TODO: Add your repository URL badge here if you have one -->
<!-- [![GitHub stars](https://img.shields.io/github/stars/your_username/sff.svg?style=social)](https://github.com/your_username/sff) -->

**sff (SemanticFileFinder)** is a command-line tool that rapidly searches for files in a given directory based on the semantic meaning of your query. It leverages sentence embeddings through `model2vec-rs` to understand content, not just keywords. It reads `.txt`, `.md`, and `.mdx` files, chunks their content, and ranks them by similarity to find the most relevant text snippets.

## Features

*   **Semantic Search:** Finds files based on meaning, not just exact keyword matches.
*   **Supported Files:** Scans `.txt`, `.md`, and `.mdx` files.
*   **Content Chunking:** Breaks down documents into smaller, manageable chunks for precise matching.
*   **Embedding Powered:** Uses `model2vec-rs` to generate text embeddings. Models are typically downloaded from Hugging Face Hub.
*   **Fast & Parallelized:** Utilizes Rayon for parallel processing of file discovery, embedding generation, and similarity calculation.
*   **Customizable:**
    *   Specify search directory.
    *   Define your semantic query.
    *   Choose the embedding model (Hugging Face Hub or local path).
    *   Limit the number of results.
    *   Enable recursive search through subdirectories.
*   **Verbose Mode:** Offers detailed timing information for performance analysis.
*   **Clickable File Paths:** Output paths are formatted for easy opening in most terminals.

## Installation

Once `sff` is published on crates.io, you can install it using Cargo:

```bash
cargo install sff
```

Ensure `~/.cargo/bin` is in your system's `PATH`.

## Usage

The basic command structure is:

```bash
sff [OPTIONS] <QUERY>...
```

**Examples:**

*   Search in the current directory for "machine learning techniques":
    ```bash
    sff "machine learning techniques"
    ```

*   Search recursively in `~/Documents/notes` for "project ideas for rust":
    ```bash
    sff -p ~/Documents/notes -r "project ideas for rust"
    ```

*   Use a different model and limit results to 5:
    ```bash
    sff -m "sentence-transformers/all-MiniLM-L6-v2" -l 5 "benefits of parallel computing"
    ```

**All Options:**

You can view all available options with `sff --help`:

```
sff: Fast semantic file finder

Usage: sff [OPTIONS] <QUERY>...

Arguments:
  <QUERY>...
          The semantic search query

Options:
  -p, --path <PATH>
          The directory to search in
          [default: .]

  -m, --model <MODEL>
          Model to use for embeddings, from Hugging Face Hub or local path
          [default: minishlab/potion-retrieval-32M]

  -l, --limit <LIMIT>
          Number of top results to display
          [default: 10]

  -r, --recursive
          Search recursively through all subdirectories

  -v, --verbose
          Enable verbose mode to print detailed timings for nerds

  -h, --help
          Print help (see more with '--help')

  -V, --version
          Print version
```

## Models

`sff` uses `model2vec-rs`, which typically downloads models from the [Hugging Face Hub](https://huggingface.co/models). The default model is `minishlab/potion-retrieval-32M`. You can specify any compatible sentence transformer model available on the Hub or a local path to a model. The first time you use a new model, it will be downloaded, which might take some time.

## Development

<!-- TODO: Add contribution guidelines or development setup if you plan for it -->

## License

This project is licensed under either of
* Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)
at your option.

---
Built by Dominik Weckmüller.