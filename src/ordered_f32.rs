#[derive(PartialEq)]
pub(crate) struct OrderedF32(pub(crate) f32);

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        f32::total_cmp(&self.0, &other.0)
    }
}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for OrderedF32 {}
