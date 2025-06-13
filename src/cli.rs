use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "sff", // Changed from semanticfinder
    author = "Dominik Weckmüller",
    version = "0.1.0",
    about = "sff: Fast semantic file finder", // Updated for clarity
    long_about = "sff (SemanticFileFinder) searches for files in a given directory based on the semantic meaning of a query with model2vec-rs. It reads .txt, .md, and .mdx files, chunks their content and ranks by similarity to find the most relevant text snippets."
)]
pub struct Args {
    /// The directory to search in
    #[arg(short = 'p', long, default_value = ".")]
    pub path: PathBuf,

    /// The semantic search query
    #[arg(required = true)]
    pub query: Vec<String>,

    /// Model to use for embeddings, from Hugging Face Hub or local path
    #[arg(short = 'm', long, default_value = "minishlab/potion-retrieval-32M")]
    pub model: String,

    /// Number of top results to display
    #[arg(short = 'l', long, default_value_t = 10)]
    pub limit: usize,

    /// Search recursively through all subdirectories
    #[arg(short = 'r', long)]
    pub recursive: bool,

    /// Enable verbose mode to print detailed timings for nerds
    #[arg(short = 'v', long)]
    pub verbose: bool,
}