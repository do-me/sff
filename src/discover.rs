//! Parallel file discovery.
//!
//! Directories are read concurrently on the rayon thread pool: each directory is one task, and
//! every subdirectory it contains is spawned as a new task. Symlinks are never followed (except
//! for the search root itself), hidden files are included and no ignore files are read.
//!
//! Only file names are inspected while reading a directory, so non-matching files cost no
//! allocations or syscalls. On Unix the directory is read with `readdir` directly, which avoids
//! the per-entry copy and allocation of `std::fs::read_dir`.

use glob::Pattern;
use rayon::prelude::*;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Dir,
    File,
    Other,
}

struct Walker<'a> {
    root: &'a Path,
    recursive: bool,
    extensions: &'a [String],
    exclude: &'a [Pattern],
    verbose: bool,
    found: Mutex<Vec<PathBuf>>,
}

/// Returns all files under `root` whose extension is in `extensions`, sorted by path.
///
/// Without `recursive` only files directly inside `root` are returned. Files and directories
/// matching one of the `exclude` patterns (against the path relative to `root` or the full path)
/// are skipped, and excluded directories are not descended into. If `root` is a file, it is
/// returned when its extension matches.
pub fn discover_files(
    root: &Path,
    recursive: bool,
    extensions: &[String],
    exclude: &[Pattern],
    verbose: bool,
) -> Vec<PathBuf> {
    let walker = Walker {
        root,
        recursive,
        extensions,
        exclude,
        verbose,
        found: Mutex::new(Vec::new()),
    };
    if walker.is_excluded(root) {
        return Vec::new();
    }
    match fs::metadata(root) {
        Ok(meta) if meta.is_dir() => {
            rayon::scope(|scope| walker.walk_dir(scope, root.to_path_buf()));
        }
        Ok(meta) if meta.is_file() => {
            if root
                .file_name()
                .is_some_and(|name| walker.has_extension(name))
            {
                walker.found.lock().unwrap().push(root.to_path_buf());
            }
        }
        _ => {}
    }
    let mut found = walker.found.into_inner().unwrap();
    found.par_sort_unstable();
    found
}

impl<'a> Walker<'a> {
    fn walk_dir<'s>(&'s self, scope: &rayon::Scope<'s>, dir: PathBuf) {
        let mut matches = Vec::new();
        for_each_entry(&dir, |name, kind| match kind {
            Kind::Dir => {
                let path = dir.join(name);
                if !self.is_excluded(&path) && self.recursive {
                    scope.spawn(move |scope| self.walk_dir(scope, path));
                }
            }
            Kind::File if self.has_extension(name) => {
                let path = dir.join(name);
                if !self.is_excluded(&path) {
                    matches.push(path);
                }
            }
            _ => {}
        });
        if !matches.is_empty() {
            self.found.lock().unwrap().append(&mut matches);
        }
    }

    fn has_extension(&self, name: &OsStr) -> bool {
        Path::new(name)
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|ext| self.extensions.iter().any(|e| e == ext))
    }

    fn is_excluded(&self, path: &Path) -> bool {
        if self.exclude.is_empty() {
            return false;
        }
        let relative = path
            .strip_prefix(self.root)
            .unwrap_or(path)
            .to_string_lossy();
        let full = path.to_string_lossy();
        let excluded = self
            .exclude
            .iter()
            .any(|p| p.matches(&relative) || p.matches(&full));
        if excluded && self.verbose {
            eprintln!("[VERBOSE] Excluding: {}", relative);
        }
        excluded
    }
}

/// Calls `f` with the name and type of every entry in `dir`, skipping `.` and `..`.
/// Symlinks are reported as `Kind::Other`. Unreadable directories are skipped silently.
#[cfg(unix)]
fn for_each_entry(dir: &Path, mut f: impl FnMut(&OsStr, Kind)) {
    use std::ffi::{CStr, CString};
    use std::os::unix::ffi::OsStrExt;

    struct Dir(*mut libc::DIR);
    impl Drop for Dir {
        fn drop(&mut self) {
            unsafe { libc::closedir(self.0) };
        }
    }

    let Ok(c_dir) = CString::new(dir.as_os_str().as_bytes()) else {
        return;
    };
    let handle = unsafe { libc::opendir(c_dir.as_ptr()) };
    if handle.is_null() {
        return;
    }
    let handle = Dir(handle);
    loop {
        // SAFETY: `handle` is a valid open directory stream used by this thread only, and the
        // returned entry is only read before the next `readdir` call.
        let entry = unsafe { libc::readdir(handle.0) };
        if entry.is_null() {
            break;
        }
        let entry = unsafe { &*entry };
        let name = unsafe { CStr::from_ptr(entry.d_name.as_ptr()) }.to_bytes();
        if name == b"." || name == b".." {
            continue;
        }
        let name = OsStr::from_bytes(name);
        let kind = match entry.d_type {
            libc::DT_DIR => Kind::Dir,
            libc::DT_REG => Kind::File,
            libc::DT_UNKNOWN => kind_of(&dir.join(name)),
            _ => Kind::Other,
        };
        f(name, kind);
    }
}

#[cfg(not(unix))]
fn for_each_entry(dir: &Path, f: impl FnMut(&OsStr, Kind)) {
    for_each_entry_std(dir, f)
}

#[cfg_attr(unix, allow(dead_code))]
fn for_each_entry_std(dir: &Path, mut f: impl FnMut(&OsStr, Kind)) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let kind = match entry.file_type() {
            Ok(t) if t.is_dir() => Kind::Dir,
            Ok(t) if t.is_file() => Kind::File,
            _ => Kind::Other,
        };
        f(&entry.file_name(), kind);
    }
}

#[cfg_attr(not(unix), allow(dead_code))]
fn kind_of(path: &Path) -> Kind {
    match fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() => Kind::Dir,
        Ok(m) if m.is_file() => Kind::File,
        _ => Kind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn exts(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn patterns(list: &[&str]) -> Vec<Pattern> {
        list.iter().map(|p| Pattern::new(p).unwrap()).collect()
    }

    /// The file discovery of sff <= 0.5.0, used as the reference behavior.
    fn walkdir_reference(
        root: &Path,
        recursive: bool,
        extensions: &[String],
        exclude: &[Pattern],
    ) -> Vec<PathBuf> {
        let mut found: Vec<PathBuf> = walkdir::WalkDir::new(root)
            .max_depth(if recursive { usize::MAX } else { 1 })
            .into_iter()
            .filter_entry(|entry| {
                let relative = entry
                    .path()
                    .strip_prefix(root)
                    .unwrap_or(entry.path())
                    .to_string_lossy();
                !exclude
                    .iter()
                    .any(|p| p.matches(&relative) || p.matches(&entry.path().to_string_lossy()))
            })
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|ext| extensions.contains(&ext.to_string()))
            })
            .map(|e| e.into_path())
            .collect();
        found.sort();
        found
    }

    fn touch(root: &Path, rel: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "some words").unwrap();
    }

    fn sample_tree() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for rel in [
            "a.txt",
            "b.md",
            "c.jpg",
            "noext",
            "upper.TXT",
            "dot.",
            ".txt",
            ".hidden.md",
            "secret.txt",
            "with space.org",
            "sub/d.txt",
            "sub/deep/e.mdx",
            "sub/deep/f.org",
            "sub/deep/g.png",
            "sub/build/h.txt",
            "sub/deep/build/i.md",
            "node_modules/pkg/j.md",
            ".git/k.txt",
            "dir.txt/l.md",
        ] {
            touch(root, rel);
        }
        fs::create_dir_all(root.join("empty/nested")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(root.join("sub"), root.join("link_dir")).unwrap();
            symlink(root.join("a.txt"), root.join("link_file.txt")).unwrap();
            symlink(root.join("missing"), root.join("dangling.txt")).unwrap();
        }
        tmp
    }

    #[test]
    fn finds_expected_files() {
        let tmp = sample_tree();
        let root = tmp.path();
        let rel = |found: Vec<PathBuf>| -> Vec<String> {
            found
                .iter()
                .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().into_owned())
                .collect()
        };
        let default = exts(&["txt", "md", "mdx", "org"]);

        assert_eq!(
            rel(discover_files(root, false, &default, &[], false)),
            [
                ".hidden.md",
                "a.txt",
                "b.md",
                "secret.txt",
                "with space.org"
            ]
        );
        assert_eq!(
            rel(discover_files(
                root,
                true,
                &default,
                &patterns(&["node_modules", "**/build", "secret.txt"]),
                false
            )),
            [
                ".git/k.txt",
                ".hidden.md",
                "a.txt",
                "b.md",
                "dir.txt/l.md",
                "sub/d.txt",
                "sub/deep/e.mdx",
                "sub/deep/f.org",
                "with space.org",
            ]
        );
    }

    #[test]
    fn matches_walkdir_reference() {
        let tmp = sample_tree();
        let root = tmp.path();
        let with_slash = PathBuf::from(format!("{}/", root.display()));
        let extension_sets = [
            exts(&["txt", "md", "mdx", "org"]),
            exts(&["md"]),
            exts(&["TXT", "jpg"]),
            exts(&[""]),
            exts(&[]),
        ];
        let pattern_sets = [
            patterns(&[]),
            patterns(&["node_modules", "**/build", "secret.txt"]),
            patterns(&["sub"]),
            patterns(&["sub/deep"]),
            patterns(&["*.md"]),
            patterns(&["**/.*"]),
            patterns(&[&format!("{}/sub/*", root.display())]),
            patterns(&["*"]),
        ];
        for search_root in [
            root.to_path_buf(),
            with_slash,
            root.join("sub"),
            root.join("a.txt"),
        ] {
            for recursive in [false, true] {
                for extensions in &extension_sets {
                    for exclude in &pattern_sets {
                        assert_eq!(
                            discover_files(&search_root, recursive, extensions, exclude, false),
                            walkdir_reference(&search_root, recursive, extensions, exclude),
                            "root={search_root:?} recursive={recursive} extensions={extensions:?} exclude={exclude:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn matches_walkdir_reference_on_large_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for i in 0..40 {
            for j in 0..5 {
                for k in 0..10 {
                    let ext = ["txt", "md", "rs", "json"][(i + j + k) % 4];
                    touch(root, &format!("d{i}/e{j}/f{k}.{ext}"));
                }
            }
            touch(root, &format!("top{i}.md"));
        }
        let extensions = exts(&["txt", "md"]);
        for exclude in [patterns(&[]), patterns(&["d1*", "*/e3"])] {
            for recursive in [false, true] {
                let found = discover_files(root, recursive, &extensions, &exclude, false);
                assert_eq!(
                    found,
                    walkdir_reference(root, recursive, &extensions, &exclude)
                );
                assert!(!found.is_empty());
            }
        }
    }

    #[test]
    fn missing_root_finds_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("missing");
        assert!(discover_files(&missing, true, &exts(&["txt"]), &[], false).is_empty());
    }

    #[test]
    fn readdir_matches_std_read_dir() {
        let tmp = sample_tree();
        let list = |read: &dyn Fn(&Path, &mut dyn FnMut(&OsStr, Kind))| {
            let mut entries = BTreeSet::new();
            read(tmp.path(), &mut |name, kind| {
                entries.insert((name.to_owned(), format!("{kind:?}")));
            });
            entries
        };
        let fast = list(&|dir, f| for_each_entry(dir, f));
        let std = list(&|dir, f| for_each_entry_std(dir, f));
        assert_eq!(fast, std);
        assert!(fast.contains(&("sub".into(), "Dir".into())));
        assert!(fast.contains(&("a.txt".into(), "File".into())));
    }
}
