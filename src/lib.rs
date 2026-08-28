mod serde_arrays;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub type EvolutionMatrix<const INPUT: usize, const OUTPUT: usize> = nalgebra::Matrix<
    f32,
    nalgebra::Const<INPUT>,
    nalgebra::Const<OUTPUT>,
    nalgebra::ArrayStorage<f32, INPUT, OUTPUT>,
>;

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Individual<
    const INPUTS: usize,
    const HIDDEN_LAYERS: usize,
    const OUTPUTS: usize,
    const NODES_FOR_HIDDEN_LAYER: usize,
> {
    pub input_matrix: EvolutionMatrix<INPUTS, NODES_FOR_HIDDEN_LAYER>,

    #[serde(with = "serde_arrays")]
    pub matricies: [EvolutionMatrix<NODES_FOR_HIDDEN_LAYER, NODES_FOR_HIDDEN_LAYER>; HIDDEN_LAYERS],
    pub output_matrix: EvolutionMatrix<NODES_FOR_HIDDEN_LAYER, OUTPUTS>,
    pub fitness: f32,
}

fn matrix_similarity<const INPUT: usize, const OUTPUT: usize>(
    one: &EvolutionMatrix<INPUT, OUTPUT>,
    two: &EvolutionMatrix<INPUT, OUTPUT>,
) -> f32 {
    let mut out = 0.0;
    for i in 0..INPUT {
        for j in 0..OUTPUT {
            out += (one[(i, j)] - two[(i, j)]) * (one[(i, j)] - two[(i, j)])
        }
    }
    return out;
}

fn relu<const SIZE: usize>(matrix: EvolutionMatrix<1, SIZE>) -> EvolutionMatrix<1, SIZE> {
    matrix.map(|k| if k > 0.0 { k } else { k / 2.0 })
}

fn sigmoid<const SIZE: usize>(matrix: EvolutionMatrix<1, SIZE>) -> EvolutionMatrix<1, SIZE> {
    matrix.map(|k| 1.0 / (1.0 + k.exp()))
}

fn random_matrix<const INPUT: usize, const OUTPUT: usize>(
    rng: &mut impl Rng,
) -> EvolutionMatrix<INPUT, OUTPUT> {
    let mut matrix = EvolutionMatrix::<INPUT, OUTPUT>::zeros();

    for i in 0..INPUT {
        for j in 0..OUTPUT {
            if rng.random_bool(0.25) {
                matrix[(i, j)] = rng.random_range(-1.0..1.0);
            }
        }
    }

    matrix
}

fn mutate_matrix<const INPUT: usize, const OUTPUT: usize>(
    matrix: &mut EvolutionMatrix<INPUT, OUTPUT>,
    rng: &mut impl Rng,
) {
    let mut attempts = 0;

    let mut x = rng.random_range(0..INPUT);
    let mut y = rng.random_range(0..OUTPUT);
    while matrix[(x, y)] == 0.0 && attempts < 3 {
        x = rng.random_range(0..INPUT);
        y = rng.random_range(0..OUTPUT);
        attempts += 1;
    }

    matrix[(x, y)] += rng.random_range(-0.1..0.1);
}

impl<const INPUTS: usize, const LAYERS: usize, const OUTPUTS: usize, const SUBLAYERS: usize>
    Individual<INPUTS, LAYERS, OUTPUTS, SUBLAYERS>
{
    pub fn evaluate(&self, inputs: [f32; INPUTS]) -> [f32; OUTPUTS] {
        let layer_1 = EvolutionMatrix::<1, INPUTS>::from_row_slice(&inputs);
        let layer_2 = relu(layer_1 * self.input_matrix);
        let layer_3 = self.matricies.iter().fold(layer_2, |a, b| relu(a * b));
        return sigmoid(layer_3 * self.output_matrix).transpose().data.0[0];
    }

    pub fn similarity(&self, other: &Self) -> f32 {
        return matrix_similarity(&self.input_matrix, &other.input_matrix)
            + self
                .matricies
                .iter()
                .zip(other.matricies.iter())
                .map(|(a, b)| matrix_similarity(a, b))
                .sum::<f32>()
            + matrix_similarity(&self.output_matrix, &other.output_matrix);
    }

    pub fn mutate(&mut self, rng: &mut impl Rng) {
        let layer = rng.random_range(0..=LAYERS + 1);
        if layer == 0 {
            mutate_matrix(&mut self.input_matrix, rng);
        } else if layer == LAYERS + 1 {
            mutate_matrix(&mut self.output_matrix, rng);
        } else {
            mutate_matrix(&mut self.matricies[layer - 1], rng);
        }
    }

    pub fn new_random(rng: &mut impl Rng) -> Self {
        let mut matricies = [EvolutionMatrix::<SUBLAYERS, SUBLAYERS>::zeros(); LAYERS];

        for i in matricies.iter_mut() {
            *i = random_matrix(rng);
        }

        return Self {
            input_matrix: random_matrix(rng),
            output_matrix: random_matrix(rng),
            matricies,
            fitness: 0.0,
        };
    }
}

#[derive(Clone, PartialEq, Debug, Serialize)]
pub struct Population<
    const INPUTS: usize,
    const LAYERS: usize,
    const OUTPUTS: usize,
    const SUBLAYERS: usize,
> {
    pub individuals: Vec<Individual<INPUTS, LAYERS, OUTPUTS, SUBLAYERS>>,
    max_size: usize,
    max_species: usize,
}

impl<const INPUTS: usize, const LAYERS: usize, const OUTPUTS: usize, const SUBLAYERS: usize>
    Population<INPUTS, LAYERS, OUTPUTS, SUBLAYERS>
{
    pub fn new(max_size: usize, max_species: usize, rng: &mut impl Rng) -> Self {
        Self {
            max_size,
            max_species,
            individuals: (0..max_size).map(|_| Individual::new_random(rng)).collect(),
        }
    }

    pub fn new_from_individuals(
        max_species: usize,
        individuals: Vec<Individual<INPUTS, LAYERS, OUTPUTS, SUBLAYERS>>,
    ) -> Self {
        Self {
            max_size: individuals.len(),
            max_species,
            individuals,
        }
    }

    /**
     * Assumes the list of species is sorted by fitness
     */
    fn speciate(&self) -> Vec<Vec<Individual<INPUTS, LAYERS, OUTPUTS, SUBLAYERS>>> {
        let mut species = vec![vec![self.individuals[0].clone()]];
        for individual in &self.individuals[1..] {
            match species
                .iter_mut()
                .find(|k| k[0].similarity(individual) < 0.5)
            {
                Some(specie) => specie.push(individual.clone()),
                None => species.push(vec![individual.clone()]),
            }
        }
        species
    }

    pub fn evolve(&mut self, rng: &mut impl Rng) {
        self.individuals
            .sort_by(|i, j| (-i.fitness).total_cmp(&-j.fitness));

        let species = self.speciate();
        let mut new_individuals: Vec<_> = species
            .iter()
            .take(self.max_species)
            .map(|i| i[0].clone())
            .collect();

        let mut i = 0;
        while new_individuals.len() < self.max_size {
            let specie = &species[i % species.len().min(self.max_species)];
            let idx1 = rng.random_range(0..specie.len());
            let idx2 = rng.random_range(0..specie.len());
            let mut new_individual = specie[idx1.min(idx2)].clone();

            for _ in 0..rng.random_range(1..=19) {
                new_individual.mutate(rng);
            }
            new_individuals.push(new_individual);
            i += 1;
        }

        for individual in new_individuals.iter_mut() {
            individual.fitness = 0.0;
        }
        self.individuals = new_individuals;
    }
}
