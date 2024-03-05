use std::sync::Mutex;

use once_cell::sync::Lazy;

struct IdGenerator {
    next_id: usize,
}

impl IdGenerator {
    fn new() -> Self {
        IdGenerator { next_id: 0 }
    }
}

impl Iterator for IdGenerator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.next_id;
        self.next_id += 1;
        Some(v)
    }
}

static ID_GEN: Lazy<Mutex<IdGenerator>> = Lazy::new(|| Mutex::new(IdGenerator::new()));

pub(crate) fn generate_id() -> usize {
    ID_GEN.lock().unwrap().next().unwrap()
}
