use evolution_rust::{Brain, Individual, MatrixBrain, initialize_random_population, mutate};
use serde_json;
use std::io::Write;

fn main() {
    let mut rng = rand::rng();
    let (mut persistant_info, mut population): (_, Vec<Brain<MatrixBrain<4, 1, 1, 4>>>) =
        initialize_random_population(100, 10, 10, &mut rng);

    let mut champions: Vec<Individual<MatrixBrain<4, 1, 1, 4>>> = Vec::new();

    let math_problems = [
        ([0.0, 0.0, 0.0, 1.0], 0.0),
        ([1.0, 0.0, 0.0, 1.0], 1.0),
        ([0.0, 1.0, 0.0, 1.0], 1.0),
        ([0.5, 0.0, 0.25, 1.0], 0.75),
        ([0.25, 0.25, 0.25, 1.0], 0.75),
        ([0.1, 0.1, 0.1, 1.0], 0.1),
    ];

    for i in 0..1600 {
        let mut individuals = population
            .into_iter()
            .map(|i| Individual {
                fitness: math_problems
                    .iter()
                    .map(|(input, solution)| {
                        -(i.get_inner().evaluate(*input)[0] - solution).powi(2)
                    })
                    .sum::<f32>(),

                brain: i,
            })
            .collect::<Vec<_>>();

        individuals.sort_by(|a, b| f32::total_cmp(&a.fitness, &b.fitness));

        if i % 4 == 0 {
            champions.push(individuals[0].clone());
            println!(
                "Generation {i:?} current fitness: {:?}",
                individuals[0].fitness
            );
        }

        population = mutate(&mut rng, &mut persistant_info, individuals);
    }

    let string = serde_json::to_string_pretty(&champions).unwrap();
    {
        let mut file = std::fs::File::create("champions.json").unwrap();
        write!(file, "{string}").unwrap();
    }
}
