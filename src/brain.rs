use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{BrainId, Evolvable, SpeciesId};

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Brain<T: Evolvable> {
    inner: T,
    pub(crate) species_id: SpeciesId,
    pub(crate) brain_id: BrainId,
}

impl<T: Evolvable> Brain<T> {
    pub(crate) fn new_random(rng: &mut impl Rng, species_id: SpeciesId, brain_id: BrainId) -> Self {
        Self {
            inner: T::new_random(rng),
            species_id: species_id,
            brain_id: brain_id,
        }
    }

    pub(crate) fn mutate(&mut self, rng: &mut impl Rng) {
        self.inner.mutate(rng);
    }

    pub(crate) fn difference(&self, other: &Self) -> f32 {
        self.inner.difference(&other.inner)
    }

    pub fn get_inner(&self) -> &T {
        &self.inner
    }
}
