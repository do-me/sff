mod cli;

use crate::cli::Args;

use anyhow::{bail, Context, Result};
use clap::Parser;
use comfy_table::{presets::UTF8_FULL, Cell, ContentArrangement, Table};
use glob::Pattern;
use indicatif::{ProgressBar, ProgressStyle};
use model2vec_rs::model::StaticModel; // Using the provided model2vec-rs
use ndarray::{Array1, ArrayView1};
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use unicode_categories::UnicodeCategories;
use walkdir::WalkDir;

const CHUNK_EMBEDDING_BATCH_SIZE: usize = 128; // How many text chunks to embed in one go per parallel task
const WORD_CHUNK_SIZE: usize = 20; // How many words per text chunk

/// A loaded model, plus a text rewriter for tokenizers that model2vec-rs can't run as-is.
///
/// model2vec-rs 0.5 ships `tokenizers` 0.21, which can't parse the `FixedLength` pre-tokenizer
/// used by e.g. `minishlab/blink-multilingual-64M`. For such models sff removes it from the
/// tokenizer and rewrites the input text instead, see [`TextRewriter`].
struct Embedder {
    model: StaticModel,
    rewriter: Option<TextRewriter>,
}

/// Rewrites text so that a tokenizer without `FixedLength` produces the original token ids.
///
/// - Overlong words get a space after every `length` chars, like `FixedLength`.
/// - Added tokens are replaced with placeholder words that are in the vocab under the added
///   token's id. model2vec-rs then uses its fast WordPiece tokenizer, which otherwise needs
///   seconds to load blink-multilingual's 30k added tokens and is slow on CJK text.
struct TextRewriter {
    length: usize,
    /// Added tokens, matched on the raw text, and their replacements.
    added_tokens: HashMap<String, String>,
    added_token_starts: HashSet<char>,
    max_added_token_chars: usize,
    /// Placeholder chars aren't in the original vocab, so in the input text they are replaced
    /// with `escape`, which isn't either and so also tokenizes to `[UNK]`.
    placeholder_chars: HashSet<char>,
    escape: char,
}

impl Embedder {
    /// Loads from the Hugging Face Hub or a local path, with L2-normalized output.
    fn load(model: &str) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            let path = if Path::new(model).exists() {
                Path::new(model).join(name)
            } else {
                hf_hub::api::sync::Api::new()?.model(model.to_string()).get(name)?
            };
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))
        };

        if let Ok(bytes) = read("tokenizer.json") {
            if String::from_utf8_lossy(&bytes).contains("\"FixedLength\"") {
                let mut tokenizer: serde_json::Value = serde_json::from_slice(&bytes)?;
                if let Some(length) = take_fixed_length(&mut tokenizer["pre_tokenizer"]) {
                    let rewriter = TextRewriter::new(&mut tokenizer, length)?;
                    let model = StaticModel::from_bytes(
                        serde_json::to_vec(&tokenizer)?,
                        read("model.safetensors")?,
                        read("config.json")?,
                        Some(true),
                    )?;
                    return Ok(Self { model, rewriter: Some(rewriter) });
                }
            }
        }

        let model = StaticModel::from_pretrained(model, None, Some(true), None)?;
        Ok(Self { model, rewriter: None })
    }

    fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {
        match &self.rewriter {
            None => self.model.encode(texts),
            Some(rw) => {
                let texts: Vec<String> = texts.iter().map(|t| rw.rewrite(t).into_owned()).collect();
                self.model.encode(&texts)
            }
        }
    }
}

impl TextRewriter {
    /// Moves the added tokens that aren't in the vocab into it, under placeholder words.
    fn new(tokenizer: &mut serde_json::Value, length: usize) -> Result<Self> {
        let added = tokenizer["added_tokens"].as_array().cloned().unwrap_or_default();
        let vocab = tokenizer["model"]["vocab"]
            .as_object_mut()
            .context("unsupported tokenizer: no vocab")?;
        if added.iter().any(|t| ["normalized", "single_word", "lstrip", "rstrip"].iter().any(|k| t[k] == true)) {
            bail!("unsupported tokenizer: added tokens with normalized, single_word, lstrip or rstrip");
        }

        // Yi syllables: letters that survive normalization unchanged and are rare in text.
        let vocab_chars: HashSet<char> = vocab.keys().flat_map(|k| k.chars()).collect();
        let mut alphabet: Vec<char> =
            ('\u{A000}'..='\u{A48C}').filter(|c| !vocab_chars.contains(c)).collect();
        let escape = alphabet.pop().context("unsupported tokenizer: no placeholder chars")?;
        if alphabet.len() < 2 {
            bail!("unsupported tokenizer: too few placeholder chars");
        }
        let mut width = 1;
        while alphabet.len().pow(width) < added.len() {
            width += 1;
        }
        let placeholder = |mut i: usize| -> String {
            let mut word = vec![alphabet[0]; width as usize];
            for c in word.iter_mut().rev() {
                *c = alphabet[i % alphabet.len()];
                i /= alphabet.len();
            }
            word.into_iter().collect()
        };

        let mut added_tokens = HashMap::new();
        let mut kept = Vec::new();
        for (i, t) in added.into_iter().enumerate() {
            let (Some(content), Some(id)) = (t["content"].as_str(), t["id"].as_u64()) else {
                bail!("unsupported tokenizer: invalid added token {t}");
            };
            if vocab.get(content).and_then(|v| v.as_u64()) == Some(id) {
                added_tokens.insert(content.to_string(), content.to_string());
                kept.push(t);
            } else {
                let word = placeholder(i);
                vocab.insert(word.clone(), id.into());
                added_tokens.insert(content.to_string(), format!(" {word} "));
            }
        }
        tokenizer["added_tokens"] = kept.into();

        Ok(Self {
            length,
            added_token_starts: added_tokens.keys().filter_map(|t| t.chars().next()).collect(),
            max_added_token_chars: added_tokens.keys().map(|t| t.chars().count()).max().unwrap_or(0),
            added_tokens,
            placeholder_chars: alphabet.into_iter().collect(),
            escape,
        })
    }

    fn rewrite<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let needs_rewrite = |c: &char| self.added_token_starts.contains(c) || self.placeholder_chars.contains(c);
        if !text.chars().any(|c| needs_rewrite(&c)) && !self.has_long_word(text) {
            return Cow::Borrowed(text);
        }
        let mut out = String::with_capacity(text.len() + text.len() / self.length);
        let mut run = 0;
        let mut rest = text;
        while let Some(c) = rest.chars().next() {
            if let Some((len, replacement)) = self.added_token_at(rest) {
                out.push_str(replacement);
                rest = &rest[len..];
                run = 0;
                continue;
            }
            rest = &rest[c.len_utf8()..];
            let c = if self.placeholder_chars.contains(&c) { self.escape } else { c };
            match char_weight(c) {
                None => run = 0,
                Some(0) => {}
                // Chars like 'İ' lowercase to several chars, which `FixedLength` may split.
                Some(_) => {
                    for lc in c.to_lowercase() {
                        if run == self.length {
                            out.push(' ');
                            run = 0;
                        }
                        out.push(lc);
                        run += 1;
                    }
                    continue;
                }
            }
            out.push(c);
        }
        Cow::Owned(out)
    }

    /// Whether `text` has a word longer than `length`. Added tokens only shorten words.
    fn has_long_word(&self, text: &str) -> bool {
        let mut run = 0;
        text.chars().any(|c| {
            run = char_weight(c).map_or(0, |w| run + w);
            run > self.length
        })
    }

    /// The byte length and replacement of the longest added token at the start of `s`.
    fn added_token_at(&self, s: &str) -> Option<(usize, &String)> {
        if !self.added_token_starts.contains(&s.chars().next()?) {
            return None;
        }
        let ends: Vec<usize> = s
            .char_indices()
            .skip(1)
            .map(|(i, _)| i)
            .chain(std::iter::once(s.len()))
            .take(self.max_added_token_chars)
            .collect();
        ends.into_iter().rev().find_map(|end| self.added_tokens.get(&s[..end]).map(|r| (end, r)))
    }
}

/// How many normalized chars `c` adds to the current word, or `None` if it ends the word.
/// Words are split as in BERT pre-tokenization; control chars are dropped by the normalizer.
fn char_weight(c: char) -> Option<usize> {
    if c.is_whitespace() || c.is_ascii_punctuation() || c.is_punctuation() || is_cjk_ideograph(c) {
        None
    } else if c.is_other() || c == '\u{fffd}' {
        Some(0)
    } else {
        Some(c.to_lowercase().count())
    }
}

/// Removes `FixedLength` from a pre-tokenizer (or a `Sequence` of them) and returns its length.
fn take_fixed_length(pre_tokenizer: &mut serde_json::Value) -> Option<usize> {
    let length = |p: &serde_json::Value| p["length"].as_u64().unwrap_or(5) as usize;
    if pre_tokenizer["type"] == "FixedLength" {
        let n = length(pre_tokenizer);
        *pre_tokenizer = serde_json::Value::Null;
        return Some(n);
    }
    let steps = pre_tokenizer["pretokenizers"].as_array_mut()?;
    let n = steps.iter().filter(|s| s["type"] == "FixedLength").map(length).min()?;
    steps.retain(|s| s["type"] != "FixedLength");
    if steps.len() == 1 {
        *pre_tokenizer = steps.remove(0);
    }
    Some(n)
}

fn is_cjk_ideograph(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

#[derive(Debug, Clone)]
struct TextChunk {
    path: PathBuf,
    text: String,
    line_number: usize,
    offset_in_line: usize,
}

#[derive(Serialize)]
struct SearchResult {
    score: f32,
    path: PathBuf,
    chunk: String,
    line_number: usize,
    offset_in_line: usize,
}

#[derive(Serialize)]
struct JsonOutput {
    query: String,
    total_chunks: usize,
    total_files: usize,
    search_time_ms: f64,
    results: Vec<JsonResult>,
}

#[derive(Serialize)]
struct JsonResult {
    score: f32,
    file_path: String,
    chunk: String,
    line_number: usize,
    offset_in_line: usize,
}

#[derive(Deserialize, Debug)]
struct PyProjectToml {
    tool: Option<ToolSection>,
}

#[derive(Deserialize, Debug)]
struct ToolSection {
    sff: Option<SffConfig>,
}

#[derive(Deserialize, Debug)]
struct SffConfig {
    exclude_dirs: Option<Vec<String>>,
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
        let bound_type = if is_cpu_bound {
            "CPU-bound"
        } else {
            "I/O-bound"
        };
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

fn load_exclude_dirs(search_path: &Path) -> Vec<String> {
    let mut current_dir = search_path.to_path_buf();
    
    // Walk up the directory tree looking for pyproject.toml
    loop {
        let pyproject_path = current_dir.join("pyproject.toml");
        
        if pyproject_path.exists() {
            if let Ok(contents) = fs::read_to_string(&pyproject_path) {
                if let Ok(parsed) = toml::from_str::<PyProjectToml>(&contents) {
                    if let Some(exclude_dirs) = parsed
                        .tool
                        .and_then(|t| t.sff)
                        .and_then(|s| s.exclude_dirs)
                    {
                        return exclude_dirs;
                    }
                }
            }
        }
        
        // Move up one directory
        if !current_dir.pop() {
            break;
        }
    }
    
    // No default exclude patterns - let users configure them explicitly
    vec![]
}

fn main() -> Result<()> {
    let program_total_start_time = Instant::now();
    let args = Args::parse();
    let query_string = args.query.join(" ");

    // 1. DISCOVER, READ, AND CHUNK FILES
    let (chunks, file_count) = timed_block(
        "File Discovery, Reading & Chunking",
        args.verbose,
        false,
        || {
            // Load exclude patterns from pyproject.toml
            let exclude_patterns = load_exclude_dirs(&args.path);
            let exclude_patterns: Vec<Pattern> = exclude_patterns
                .iter()
                .filter_map(|p| Pattern::new(p).ok())
                .collect();
                
            let walker = WalkDir::new(&args.path)
                .max_depth(if args.recursive { usize::MAX } else { 1 })
                .into_iter()
                .filter_entry(|entry| {
                    // Get path relative to search root
                    let relative_path = entry.path().strip_prefix(&args.path)
                        .unwrap_or(entry.path())
                        .to_string_lossy();
                    
                    // Check if any exclude pattern matches
                    let should_exclude = exclude_patterns.iter().any(|pattern| {
                        pattern.matches(&relative_path) || 
                        pattern.matches(&entry.path().to_string_lossy())
                    });
                    
                    if should_exclude && args.verbose {
                        eprintln!("[VERBOSE] Excluding: {}", relative_path);
                    }
                    
                    !should_exclude
                });
            
            let collected_chunks: Vec<TextChunk> = walker
                .filter_map(Result::ok)
                .par_bridge()
                .filter(|e| e.file_type().is_file())
                .filter_map(|entry| {
                    let path = entry.path();
                    let extension = path.extension().and_then(|s| s.to_str())?;
                    if args.extension.contains(&extension.to_string()) {
                        match fs::read_to_string(path) {
                            Ok(content) => Some((content, path.to_path_buf())),
                            Err(e) => {
                                if args.verbose {
                                    eprintln!("[VERBOSE] Failed to read {}: {}", path.display(), e);
                                }
                                None
                            }
                        }
                    } else {None}
                })
                .flat_map(|(content, path)| {
                    let mut words: Vec<(usize, usize)> = Vec::new(); // Vec<(start, end)>
                    let mut position = 0;
                    for word in content.split_whitespace() {
                        if let Some(relative_start) = content[position..].find(word) {
                            let absolute_start = position + relative_start;
                            let absolute_end = absolute_start + word.len();
                            words.push((absolute_start, absolute_end));
                            position = absolute_end;
                        }
                    }
                    let mut newline_count: usize = 0;
                    words
                        .chunks(WORD_CHUNK_SIZE)
                        .map(|word_slice| {
                            let start = word_slice.first().unwrap().0;
                            let end = word_slice.last().unwrap().1;
                            let substring = &content[start..end];

                            let newline_positions: Vec<usize> =
                                substring.match_indices('\n').map(|(idx, _)| idx).collect();
                            let first_line_of_chunk = substring.lines().next().unwrap_or("");
                            let mut line_number = 1;
                            let mut offset_in_line = 0;
                            for (idx, line) in content.lines().enumerate() {
                                if idx >= newline_count {
                                    if let Some(pos) = line.find(first_line_of_chunk) {
                                        line_number = idx + 1;
                                        offset_in_line = pos + 1;
                                        break;
                                    }
                                }
                            }

                            newline_count = newline_count + newline_positions.len();

                            TextChunk {
                                path: path.clone(),
                                text: substring.to_string(),
                                line_number: line_number,
                                offset_in_line: offset_in_line,
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            let num_unique_files = {
                let unique_paths: HashSet<_> = collected_chunks.iter().map(|c| &c.path).collect();
                unique_paths.len()
            };
            (collected_chunks, num_unique_files)
        },
    );

    if chunks.is_empty() {
        println!(
            "No text files of type {} found to search in '{}'.",
            args.extension.iter().map(|s| {
                ".".to_string() + s
            }).collect::<Vec<_>>().join(", "),
            args.path.display()
        );
        return Ok(());
    }

    // 2. LOAD MODEL
    let model = timed_block("Model Loading", args.verbose, false, || {
        Embedder::load(&args.model)
    })?;

    if args.verbose && program_total_start_time.elapsed().as_secs_f32() > 0.5 {
        eprintln!("[VERBOSE] Note: If model loading is slow (>500ms), it might be due to a first-time download by hf-hub or a large model.");
        // Changed fast_finder to sff
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

    let chunk_embeddings: Vec<Vec<f32>> =
        timed_block("Chunk Embedding Generation", args.verbose, true, || {
            chunks
                .par_chunks(CHUNK_EMBEDDING_BATCH_SIZE)
                .flat_map(|batch_of_text_chunks| {
                    let texts_for_batch: Vec<String> = batch_of_text_chunks
                        .iter()
                        .map(|tc| tc.text.clone())
                        .collect();
                    let embeddings_for_batch = model_arc.encode(&texts_for_batch);
                    bar_chunk_embedding.inc(batch_of_text_chunks.len() as u64);
                    embeddings_for_batch
                })
                .collect()
        });
    bar_chunk_embedding.finish_with_message("Done embedding chunks.");

    // 5. CALCULATE SIMILARITY AND SORT RESULTS
    let query_vec: Array1<f32> = Array1::from(query_embedding);

    let mut results: Vec<SearchResult> = timed_block(
        "Similarity Calculation & Sorting",
        args.verbose,
        true,
        || {
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
                        line_number: chunks[i].line_number.clone(),
                        offset_in_line: chunks[i].offset_in_line.clone(),
                    }
                })
                .collect();

            collected_results.par_sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            collected_results
        },
    );

    // 6. OUTPUT THE RESULTS
    if args.verbose {
        eprintln!("[VERBOSE] Result Printing Start");
    }

    let elapsed_time_total = program_total_start_time.elapsed();

    if args.json {
        // JSON output
        let json_results: Vec<JsonResult> = results
            .iter()
            .take(args.limit)
            .map(|result| JsonResult {
                score: result.score,
                file_path: result.path.to_string_lossy().to_string(),
                chunk: result.chunk.clone(),
                line_number: result.line_number,
                offset_in_line: result.offset_in_line,
            })
            .collect();

        let json_output = JsonOutput {
            query: query_string,
            total_chunks: results.len(),
            total_files: file_count,
            search_time_ms: elapsed_time_total.as_secs_f64() * 1000.0,
            results: json_results,
        };

        println!("{}", serde_json::to_string_pretty(&json_output)?);
    } else {
        // Table output (existing logic)
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
                    result.chunk = result
                        .chunk
                        .chars()
                        .take(MAX_CHUNK_DISPLAY_LEN)
                        .collect::<String>()
                        + "...";
                }
                table.add_row(vec![
                    Cell::new(format!("{:.2}", result.score)),
                    Cell::new(&result.chunk),
                    Cell::new(format_path_for_terminal(
                        &result.path,
                        &result.line_number,
                        &result.offset_in_line,
                    )),
                ]);
            }
            println!("{table}");
        } else {
            println!("No matches found.");
        }
    }

    if args.verbose {
        eprintln!(
            "[VERBOSE] Result Printing End: {:.2} ms (cumulative)",
            program_total_start_time.elapsed().as_secs_f64() * 1000.0
        );
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

fn format_path_for_terminal(path: &Path, line_number: &usize, word_offset: &usize) -> String {
    let (path_to_display, is_canonical) = match path.canonicalize() {
        Ok(abs_path) => (abs_path, true),
        Err(_) => (path.to_path_buf(), false),
    };

    let path_str = path_to_display.to_string_lossy();
    let encoded_path = utf8_percent_encode(&path_str, PATH_ENCODE_SET).to_string();

    if is_canonical && path_str.starts_with("\\\\?\\") {
        format!(
            "file:///{}:{}:{}",
            path_str.trim_start_matches("\\\\?\\").replace('\\', "/"),
            line_number,
            word_offset
        )
    } else if cfg!(windows) && is_canonical {
        format!(
            "file:///{}:{}:{}",
            encoded_path.replace('\\', "/"),
            line_number,
            word_offset
        )
    } else {
        format!("file://{}:{}:{}", encoded_path, line_number, word_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_text_for_tokenizer_without_fixed_length() {
        let mut tokenizer = serde_json::json!({
            "added_tokens": [
                {"id": 0, "content": "[UNK]", "normalized": false, "special": true},
                {"id": 3, "content": "一个", "normalized": false, "special": false},
            ],
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "BertPreTokenizer"}, {"type": "FixedLength", "length": 4},
            ]},
            "model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "a": 1, "##a": 2}},
        });
        let length = take_fixed_length(&mut tokenizer["pre_tokenizer"]).unwrap();
        assert_eq!(length, 4);
        assert_eq!(tokenizer["pre_tokenizer"], serde_json::json!({"type": "BertPreTokenizer"}));

        let rw = TextRewriter::new(&mut tokenizer, length).unwrap();
        let placeholder = tokenizer["model"]["vocab"]
            .as_object()
            .unwrap()
            .iter()
            .find(|(_, id)| **id == 3)
            .map(|(word, _)| word.clone())
            .unwrap();
        assert_eq!(tokenizer["added_tokens"].as_array().unwrap().len(), 1);

        assert_eq!(rw.rewrite("aa aa, [UNK]"), "aa aa, [UNK]");
        // 'İ' lowercases to two chars, the second of which starts a new chunk.
        assert_eq!(rw.rewrite("AAAİa"), "aaai \u{307}a");
        assert_eq!(rw.rewrite("aaa一个aaaaa"), format!("aaa {placeholder} aaaa a"));
        let yi = placeholder.chars().next().unwrap();
        assert_eq!(rw.rewrite(&format!("a{yi}")), format!("a{}", rw.escape));
    }
}
