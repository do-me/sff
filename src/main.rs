mod cli;

use crate::cli::Args;

use anyhow::{Context, Result};
use clap::Parser;
use comfy_table::{presets::UTF8_FULL, Cell, ContentArrangement, Table};
use indicatif::{ProgressBar, ProgressStyle};
use model2vec_rs::model::StaticModel; // Using the provided model2vec-rs
use ndarray::{Array1, ArrayView1};
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant};
use walkdir::WalkDir;

const CHUNK_EMBEDDING_BATCH_SIZE: usize = 128; // How many text chunks to embed in one go per parallel task
const WORD_CHUNK_SIZE: usize = 20; // How many words per text chunk

#[derive(Debug, Clone)]
struct TextChunk {
    path: PathBuf,
    text: String,
}

struct SearchResult {
    score: f32,
    path: PathBuf,
    chunk: String,
}

const PATH_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'#'); // Added '#' as it's common in file names and needs encoding for file:// URLs

// Helper for timed blocks in verbose mode
fn timed_block<F, R>(name: &str, verbose: bool, is_cpu_bound: bool, func: F) -> R
where
    F: FnOnce() -> R,
{
    if verbose {
        let start_time = Instant::now();
        let result = func();
        let duration = start_time.elapsed();
        let bound_type = if is_cpu_bound { "CPU-bound" } else { "I/O-bound" };
        eprintln!(
            "[VERBOSE] {}: {:.2} ms ({})",
            name,
            duration.as_secs_f64() * 1000.0,
            bound_type
        );
        result
    } else {
        func()
    }
}

fn main() -> Result<()> {
    let program_total_start_time = Instant::now();
    let args = Args::parse();
    let query_string = args.query.join(" ");

    // 1. DISCOVER, READ, AND CHUNK FILES
    let (chunks, file_count) =
        timed_block("File Discovery, Reading & Chunking", args.verbose, false, || {
            let walker = WalkDir::new(&args.path).max_depth(if args.recursive { usize::MAX } else { 1 });
            let collected_chunks: Vec<TextChunk> = walker
                .into_iter()
                .filter_map(Result::ok)
                .par_bridge()
                .filter(|e| e.file_type().is_file())
                .filter_map(|entry| {
                    let path = entry.path();
                    let extension = path.extension().and_then(|s| s.to_str());
                    match extension {
                        Some("txt") | Some("md") | Some("mdx") => {
                            match fs::read_to_string(path) {
                                Ok(content) => Some((content, path.to_path_buf())),
                                Err(e) => {
                                    if args.verbose {
                                        eprintln!("[VERBOSE] Failed to read {}: {}", path.display(), e);
                                    }
                                    None
                                }
                            }
                        },
                        _ => None,
                    }
                })
                .flat_map(|(content, path)| {
                    let words: Vec<&str> = content.split_whitespace().collect();
                    words
                        .chunks(WORD_CHUNK_SIZE)
                        .map(|word_slice| TextChunk {
                            path: path.clone(),
                            text: word_slice.join(" "),
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            let num_unique_files = {
                let unique_paths: HashSet<_> = collected_chunks.iter().map(|c| &c.path).collect();
                unique_paths.len()
            };
            (collected_chunks, num_unique_files)
        });

    if chunks.is_empty() {
        println!(
            "No text files (.txt, .md, .mdx) found to search in '{}'.",
            args.path.display()
        );
        return Ok(());
    }

    // 2. LOAD MODEL
    let model = timed_block("Model Loading", args.verbose, false, || {
        StaticModel::from_pretrained(&args.model, None, Some(true), None) // normalize=true
    })?;
    
    if args.verbose && program_total_start_time.elapsed().as_secs_f32() > 0.5 {
         eprintln!("[VERBOSE] Note: If model loading is slow (>500ms), it might be due to first-time download by hf-hub, or inefficiencies in the specific `model2vec-rs/model.rs::from_pretrained` version being used (e.g., for `unk_token` lookup). This part cannot be optimized further within `sff` itself without changing `model2vec-rs`."); // Changed fast_finder to sff
    }

    let model_arc = Arc::new(model);

    // 3. ENCODE THE SEARCH QUERY
    let query_embedding = timed_block("Query Embedding", args.verbose, true, || {
        let query_embeddings_vec = model_arc.encode(&[query_string.clone()]);
        query_embeddings_vec
            .into_iter()
            .next()
            .context("Failed to encode query string")
    })?;

    // 4. GENERATE EMBEDDINGS FOR TEXT CHUNKS
    let bar_chunk_embedding = ProgressBar::new(chunks.len() as u64);
    if args.verbose || chunks.len() > 10000 {
        bar_chunk_embedding.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
                .progress_chars("#>-"),
        );
    } else {
        bar_chunk_embedding.set_style(ProgressStyle::default_bar().template("")?);
    }
    bar_chunk_embedding.set_message("Embedding file chunks...");

    let chunk_embeddings: Vec<Vec<f32>> = timed_block("Chunk Embedding Generation", args.verbose, true, || {
        chunks
            .par_chunks(CHUNK_EMBEDDING_BATCH_SIZE)
            .flat_map(|batch_of_text_chunks| {
                let texts_for_batch: Vec<String> = batch_of_text_chunks.iter().map(|tc| tc.text.clone()).collect();
                let embeddings_for_batch = model_arc.encode(&texts_for_batch);
                bar_chunk_embedding.inc(batch_of_text_chunks.len() as u64);
                embeddings_for_batch
            })
            .collect()
    });
    bar_chunk_embedding.finish_with_message("Done embedding chunks.");

    // 5. CALCULATE SIMILARITY AND SORT RESULTS
    let query_vec: Array1<f32> = Array1::from(query_embedding);

    let mut results: Vec<SearchResult> = timed_block("Similarity Calculation & Sorting", args.verbose, true, || {
        let mut collected_results: Vec<SearchResult> = chunk_embeddings
            .par_iter()
            .enumerate()
            .map(|(i, emb_ref)| {
                let chunk_vec_view: ArrayView1<f32> = ArrayView1::from(emb_ref); 
                let sim = cosine_similarity(query_vec.view(), chunk_vec_view);
                SearchResult {
                    score: sim,
                    path: chunks[i].path.clone(),
                    chunk: chunks[i].text.clone(),
                }
            })
            .collect();

        collected_results.par_sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        collected_results
    });

    // 6. PRETTY-PRINT THE RESULTS
    if args.verbose {
        eprintln!("[VERBOSE] Result Printing Start");
    }

    let elapsed_time_total = program_total_start_time.elapsed();
    println!(
        "\nFound {} relevant chunks from {} files for query \"{}\" in {:.2} ms. Top {} results:",
        results.len(),
        file_count,
        query_string,
        elapsed_time_total.as_secs_f64() * 1000.0,
        args.limit.min(results.len())
    );

    if !results.is_empty() {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("Score"),
                Cell::new("Matching Text Chunk"),
                Cell::new("File Path"),
            ]);

        for result in results.iter_mut().take(args.limit) {
            const MAX_CHUNK_DISPLAY_LEN: usize = 100;
            if result.chunk.chars().count() > MAX_CHUNK_DISPLAY_LEN {
                result.chunk = result.chunk.chars().take(MAX_CHUNK_DISPLAY_LEN).collect::<String>() + "...";
            }
            table.add_row(vec![
                Cell::new(format!("{:.2}", result.score)),
                Cell::new(&result.chunk),
                Cell::new(format_path_for_terminal(&result.path)),
            ]);
        }
        println!("{table}");
    } else {
        println!("No matches found.");
    }
    
    if args.verbose {
       eprintln!("[VERBOSE] Result Printing End: {:.2} ms (cumulative)", program_total_start_time.elapsed().as_secs_f64() * 1000.0);
    }

    Ok(())
}

fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot_product = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn format_path_for_terminal(path: &Path) -> String {
    let (path_to_display, is_canonical) = match path.canonicalize() {
        Ok(abs_path) => (abs_path, true),
        Err(_) => (path.to_path_buf(), false), 
    };
    
    let path_str = path_to_display.to_string_lossy();
    let encoded_path = utf8_percent_encode(&path_str, PATH_ENCODE_SET).to_string();
    
    if is_canonical && path_str.starts_with("\\\\?\\") {
        format!("file:///{}", path_str.trim_start_matches("\\\\?\\").replace('\\', "/"))
    } else if cfg!(windows) && is_canonical {
        format!("file:///{}", encoded_path.replace('\\', "/"))
    }
    else {
         format!("file://{}", encoded_path)
    }
}