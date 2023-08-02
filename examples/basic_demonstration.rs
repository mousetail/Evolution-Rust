use evolution_rust::{Individual, Population};
use serde_json;
use std::io::Write;

fn main() {
    let mut rng = rand::thread_rng();
    let mut population: Population<4, 1, 1, 4> = Population::new(100, 20, &mut rng);

    let mut champions: Vec<Individual<4, 1, 1, 4>> = Vec::new();

    let math_problems = vec![
        ([0.0, 0.0, 0.0, 1.0], 0.0),
        ([1.0, 0.0, 0.0, 1.0], 1.0),
        ([0.0, 1.0, 0.0, 1.0], 1.0),
        ([0.5, 0.0, 0.25, 1.0], 0.75),
    ];

    for i in 0..1600 {
        for individual in population.individuals.iter_mut() {
            for (math_problem, solution) in math_problems.iter() {
                individual.fitness -=
                    (individual.evaluate(math_problem.clone())[0] - solution).powi(2);
            }
        }

        if i % 4 == 0 {
            champions.push(population.individuals[0].clone());
            println!(
                "Generation {i:?} current fitness: {:?}",
                population.individuals[0].fitness
            );
        }
        population.evolve(&mut rng);
    }

    let string = serde_json::to_string_pretty(&champions).unwrap();
    {
        let mut file = std::fs::File::create("champions.json").unwrap();
        write!(file, "{string}").unwrap();
    }
}
