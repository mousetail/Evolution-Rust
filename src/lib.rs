mod brain;
mod matrix_brain;
mod ordered_f32;
mod serde_arrays;

use std::collections::HashMap;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ordered_f32::OrderedF32;
pub use brain::Brain;
pub use matrix_brain::{EvolutionMatrix, MatrixBrain};

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Copy, Ord, PartialOrd, Eq, Hash)]
pub struct SpeciesId(u64);

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Copy, Ord, PartialOrd, Eq, Hash)]
pub struct BrainId(u64);

pub trait Evolvable: Clone {
    fn mutate(&mut self, rng: &mut impl Rng);
    fn new_random(rng: &mut impl Rng) -> Self;
    fn difference(&self, other: &Self) -> f32;
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Individual<T: Evolvable> {
    pub brain: Brain<T>,
    pub fitness: f32,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PersistentInfo {
    max_size: usize,
    max_species: usize,

    max_species_id: u64,
    max_brain_id: u64,

    round_number: u64,
    mass_extinction_timing: u64,
}

pub fn initialize_random_population<T: Evolvable>(
    max_size: usize,
    max_species: usize,
    mass_extinction_timing: u64,
    rng: &mut impl Rng,
) -> (PersistentInfo, Vec<Brain<T>>) {
    (
        PersistentInfo {
            max_size,
            max_species,
            max_species_id: 0,
            max_brain_id: 0,
            round_number: 0,
            mass_extinction_timing,
        },
        (0..max_size)
            .map(|k| Brain::new_random(rng, SpeciesId(k as u64), BrainId(k as u64)))
            .collect(),
    )
}

pub fn mutate<T: Evolvable>(
    rng: &mut impl Rng,
    persistant_info: &mut PersistentInfo,
    mut individuals: Vec<Individual<T>>,
) -> Vec<Brain<T>> {
    persistant_info.round_number += 1;
    individuals.sort_by_key(|i| OrderedF32(-i.fitness));
    let mut species_map: HashMap<SpeciesId, Vec<Individual<T>>> = HashMap::new();
    for individual in individuals {
        species_map
            .entry(individual.brain.species_id)
            .or_default()
            .push(individual.clone());
    }

    let mut species = species_map.into_iter().collect::<Vec<_>>();

    let is_mass_extinction =
        persistant_info.round_number % persistant_info.mass_extinction_timing == 0;

    if is_mass_extinction {
        species.sort_by_key(|s| OrderedF32(-s.1[0].fitness));
        species.truncate(persistant_info.max_species / 2);

        let mut new_species = vec![];

        for species in &species {
            let champion = species.1[0].clone();

            let Some(mut furthest) = species
                .1
                .iter()
                .skip(1)
                .take(species.1.len() / 2)
                .max_by_key(|i| OrderedF32(i.brain.difference(&champion.brain)))
                .cloned()
            else {
                continue;
            };

            persistant_info.max_species_id += 1;
            let species_id = SpeciesId(persistant_info.max_species_id);
            furthest.brain.species_id = species_id;
            new_species.push((species_id, vec![furthest]));
        }

        species.extend_from_slice(&new_species);
    }

    let mut new_individuals: Vec<_> = species.iter().map(|i| i.1[0].clone()).collect();

    let mut i = 0;
    while new_individuals.len() < persistant_info.max_size {
        let specie = &species[i % species.len().min(persistant_info.max_species)];
        let idx1 = rng.random_range(0..specie.1.len());
        let idx2 = rng.random_range(0..specie.1.len());
        let mut new_individual = specie.1[idx1.min(idx2)].clone();
        persistant_info.max_brain_id += 1;
        new_individual.brain.brain_id = BrainId(persistant_info.max_brain_id);

        for _ in 0..rng.random_range(1..=19).min(rng.random_range(1..19)) {
            new_individual.brain.mutate(rng);
        }
        new_individuals.push(new_individual);
        i += 1;
    }
    new_individuals.into_iter().map(|i| i.brain).collect()
}
