use clap::{Parser, ValueEnum};
use heavykeeper::{BucketedTopK, CuckooTopK, TopK};
use memmap2::Mmap;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead};
use std::process::exit;

const MAX_WORD_LEN: usize = 64;

#[derive(Debug, Clone)]
struct Word {
    bytes: [u8; MAX_WORD_LEN],
    len: u8,
}

impl Word {
    fn new() -> Self {
        Word {
            bytes: [0; MAX_WORD_LEN],
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn push(&mut self, byte: u8) {
        if self.len < MAX_WORD_LEN as u8 {
            self.bytes[self.len as usize] = byte;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

// Only hash the actual content
impl Hash for Word {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

// Compare only the actual content
impl PartialEq for Word {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Word {}

// Order by actual content
impl PartialOrd for Word {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Word {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl std::fmt::Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // we always have valid UTF-8
        let s = unsafe { std::str::from_utf8_unchecked(self.as_slice()) };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Variant {
    Topk,
    Bucketed,
    Cuckoo,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short = 'k')]
    k: usize,

    #[arg(short = 'w', default_value_t = 8192)]
    width: usize,

    #[arg(short = 'd', default_value_t = 2)]
    depth: usize,

    #[arg(short = 'y', default_value_t = 0.9)]
    decay: f64,

    #[arg(short = 'f')]
    input: Option<String>,

    /// Which sketch variant backs the top-k tracking.
    #[arg(short = 'v', long, value_enum, default_value_t = Variant::Topk)]
    variant: Variant,
}

trait Sketch {
    fn add(&mut self, word: &Word, increment: u64);
    fn print_results(&self);
}

impl Sketch for TopK<Word> {
    fn add(&mut self, word: &Word, increment: u64) {
        TopK::add(self, word, increment);
    }
    fn print_results(&self) {
        for node in self.list() {
            println!("{} {}", node.item, node.count);
        }
    }
}

impl Sketch for BucketedTopK<Word> {
    fn add(&mut self, word: &Word, increment: u64) {
        BucketedTopK::add(self, word, increment);
    }
    fn print_results(&self) {
        for node in self.list() {
            println!("{} {}", node.item, node.count);
        }
    }
}

impl Sketch for CuckooTopK<Word> {
    fn add(&mut self, word: &Word, increment: u64) {
        CuckooTopK::add(self, word, increment);
    }
    fn print_results(&self) {
        for node in self.list() {
            println!("{} {}", node.item, node.count);
        }
    }
}

fn main() {
    let args = Args::parse();

    let mut sketch: Box<dyn Sketch> = match args.variant {
        Variant::Topk => Box::new(TopK::<Word>::new(args.k, args.width, args.depth, args.decay)),
        Variant::Bucketed => Box::new(BucketedTopK::<Word>::new(
            args.k, args.width, args.depth, args.decay,
        )),
        Variant::Cuckoo => Box::new(CuckooTopK::<Word>::new(
            args.k, args.width, args.depth, args.decay,
        )),
    };
    let mut word = Word::new();

    if args.input.is_none() {
        let stdin = io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut buffer = Vec::with_capacity(1024 * 1024);

        while stdin_lock.read_until(b'\n', &mut buffer).unwrap() > 0 {
            process_bytes(&buffer, sketch.as_mut(), &mut word);
            buffer.clear();
        }
    } else {
        let file = std::fs::File::open(args.input.unwrap()).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            exit(1);
        });

        let mmap = unsafe { Mmap::map(&file) }.unwrap_or_else(|e| {
            eprintln!("Error mapping file: {}", e);
            exit(1);
        });

        process_bytes(&mmap, sketch.as_mut(), &mut word);
    }

    sketch.print_results();
}

fn process_bytes(bytes: &[u8], sketch: &mut dyn Sketch, word: &mut Word) {
    let mut pos = 0;
    let len = bytes.len();

    while pos < len {
        // Skip non-alphabetic characters
        while pos < len && !bytes[pos].is_ascii_alphabetic() {
            pos += 1;
        }

        if pos >= len {
            break;
        }

        // Find end of word
        let word_start = pos;
        while pos < len && bytes[pos].is_ascii_alphabetic() {
            pos += 1;
        }

        let word_len = pos - word_start;
        if word_len > 0 && word_len <= MAX_WORD_LEN {
            // Clear and reuse the word
            word.clear();

            // Convert to lowercase while copying
            for &b in &bytes[word_start..pos] {
                word.push(b.to_ascii_lowercase());
            }

            // Add to the sketch
            sketch.add(word, 1);
        }
    }
}
