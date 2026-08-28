use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Evolvable, serde_arrays};

pub type EvolutionMatrix<const INPUT: usize, const OUTPUT: usize> = nalgebra::Matrix<
    f32,
    nalgebra::Const<INPUT>,
    nalgebra::Const<OUTPUT>,
    nalgebra::ArrayStorage<f32, INPUT, OUTPUT>,
>;

pub(crate) fn matrix_difference<const INPUT: usize, const OUTPUT: usize>(
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

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct Layer<const SIZE: usize> {
    matrix: EvolutionMatrix<SIZE, SIZE>,
    enabled: bool,
    activation_strengh: f32,
}

impl<const SIZE: usize> Layer<SIZE> {
    pub fn difference(&self, other: &Layer<SIZE>) -> f32 {
        if !self.enabled && !other.enabled {
            return 0.0;
        }

        let identity = EvolutionMatrix::identity();

        return matrix_difference(
            if self.enabled {
                &self.matrix
            } else {
                &identity
            },
            if other.enabled {
                &other.matrix
            } else {
                &identity
            },
        );
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct MatrixBrain<
    const INPUTS: usize,
    const HIDDEN_LAYERS: usize,
    const OUTPUTS: usize,
    const NODES_FOR_HIDDEN_LAYER: usize,
> {
    input_matrix: EvolutionMatrix<INPUTS, NODES_FOR_HIDDEN_LAYER>,
    input_activation_strength: f32,

    #[serde(with = "serde_arrays")]
    matricies: [Layer<NODES_FOR_HIDDEN_LAYER>; HIDDEN_LAYERS],
    output_matrix: EvolutionMatrix<NODES_FOR_HIDDEN_LAYER, OUTPUTS>,
}

fn relu<const SIZE: usize>(
    matrix: EvolutionMatrix<1, SIZE>,
    strength: f32,
) -> EvolutionMatrix<1, SIZE> {
    matrix.map(|k| if k > 0.0 { k } else { k * strength })
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
    MatrixBrain<INPUTS, LAYERS, OUTPUTS, SUBLAYERS>
{
    pub fn evaluate(&self, inputs: [f32; INPUTS]) -> [f32; OUTPUTS] {
        let layer_1 = EvolutionMatrix::<1, INPUTS>::from_row_slice(&inputs);
        let layer_2 = relu(layer_1 * self.input_matrix, self.input_activation_strength);
        let layer_3 = self.matricies.iter().fold(layer_2, |a, b| {
            if b.enabled {
                relu(a * b.matrix, b.activation_strengh)
            } else {
                a
            }
        });
        return sigmoid(layer_3 * self.output_matrix).transpose().data.0[0];
    }
}

impl<const INPUTS: usize, const LAYERS: usize, const OUTPUTS: usize, const SUBLAYERS: usize>
    Evolvable for MatrixBrain<INPUTS, LAYERS, OUTPUTS, SUBLAYERS>
{
    fn mutate(&mut self, rng: &mut impl Rng) {
        let layer = rng.random_range(0..=LAYERS + 1);
        if layer == 0 {
            mutate_matrix(&mut self.input_matrix, rng);
        } else if layer == LAYERS + 1 {
            mutate_matrix(&mut self.output_matrix, rng);
        } else {
            mutate_matrix(&mut self.matricies[layer - 1].matrix, rng);
        }
    }

    fn difference(&self, other: &Self) -> f32 {
        return matrix_difference(&self.input_matrix, &other.input_matrix)
            + self
                .matricies
                .iter()
                .zip(other.matricies.iter())
                .map(|(a, b)| a.difference(b))
                .sum::<f32>()
            + matrix_difference(&self.output_matrix, &other.output_matrix);
    }

    fn new_random(rng: &mut impl Rng) -> Self {
        let matricies = [Layer {
            matrix: EvolutionMatrix::<SUBLAYERS, SUBLAYERS>::identity(),
            enabled: false,
            activation_strengh: 0.2,
        }; LAYERS];

        return Self {
            input_matrix: random_matrix(rng),
            output_matrix: random_matrix(rng),
            matricies,
            input_activation_strength: 0.2,
        };
    }
}
