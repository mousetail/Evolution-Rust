use std::f32::consts::PI;

use evolution_rust::{Individual, MatrixBrain};
use ggez::{
    conf::WindowSetup,
    graphics::{Color, TextAlign, TextLayout},
    *,
};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

type Brain = evolution_rust::Brain<MatrixBrain<10, 2, 4, 5>>;

const NUMBER_OF_FOOD_LOCATIONS: usize = 168;
const NUMBER_OF_WALL_LOCATIONS: usize = 101;
const ARENA_SIZE: f32 = 50.0;
const WALL_RADIUS: f32 = 3.0;
const FOOD_RADIUS: f32 = 0.5;
const NUMBER_OF_LAYERS: usize = 5;

const INPUT_LABELS: &[&str] = &[
    "food",
    "food v delta",
    "food h delta",
    "v velocity",
    "h velocity",
    "wall",
    "wall v delta",
    "wall h delta",
    "angular velocity",
    "bias",
];

const OUTPUT_LABELS: &[&str] = &["forward", "back", "left", "right"];

#[derive(Serialize, Deserialize)]
struct Spaceship {
    brain_index: usize,
    food: Vec<bool>,
    almost_food: Vec<bool>,
    location: glam::Vec2,
    angle: f32,
    alive: bool,
    velocity: glam::Vec2,
    angular_velocity: f32,

    time_alive: u32,
    time_last_food_eaten: u32,
}

impl Spaceship {
    const MASS: f32 = 4.0;
    const MOMENT_OF_INERTIA: f32 = 16.0;

    fn new(brain_index: usize, angle: f32) -> Self {
        Spaceship {
            brain_index,
            food: vec![true; NUMBER_OF_FOOD_LOCATIONS],
            almost_food: vec![true; NUMBER_OF_FOOD_LOCATIONS],
            angle,
            location: glam::Vec2::new(0., 0.),
            alive: true,
            velocity: glam::Vec2::new(0., 0.),
            angular_velocity: 0.,

            time_alive: 0,
            time_last_food_eaten: 0,
        }
    }

    fn apply_force(&mut self, force: glam::Vec2, position: glam::Vec2) {
        self.angular_velocity +=
            glam::Vec2::new(position.y, position.x).dot(force) / Self::MOMENT_OF_INERTIA;
        self.velocity += force.rotate(glam::Vec2::from_angle(self.angle)) / Self::MASS;
    }

    fn die(&mut self, time: u32) {
        self.alive = false;
        self.time_alive = time;
    }

    fn eat(&mut self, food_index: usize, time: u32) {
        self.time_last_food_eaten = time;
        self.food[food_index] = false;
    }
}

fn weight_to_color(weight: f32) -> graphics::Color {
    let value = weight.abs();
    let value_u8 = (value * 127.0) as u8;
    if weight > 0.0 {
        graphics::Color::from_rgb(
            128u8.saturating_sub(value_u8),
            128u8.saturating_add(value_u8),
            128u8.saturating_sub(value_u8),
        )
    } else {
        graphics::Color::from_rgb(
            128u8.saturating_add(value_u8),
            128u8.saturating_sub(value_u8),
            128u8.saturating_sub(value_u8),
        )
    }
}

struct State {
    spaceships: Vec<Spaceship>,
    brains: Vec<Brain>,
    steps: u32,
    round: u32,
    best_fitness: f32,
    food_eaten: Vec<u32>,

    persistant_info: evolution_rust::PersistentInfo,

    wall_locations: Vec<glam::Vec2>,
    food_locations: Vec<glam::Vec2>,

    last_input_weights: [f32; 10],
    last_output_weights: [f32; 4],
}

impl State {
    fn draw_matrix<const WIDTH: usize, const HEIGHT: usize>(
        matrix: evolution_rust::EvolutionMatrix<WIDTH, HEIGHT>,
        canvas: &mut graphics::Canvas,
        ctx: &mut Context,
        starting_height: usize,
        rect: graphics::Rect,
    ) -> GameResult {
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                if matrix[(i, j)] != 0.0 {
                    let color = weight_to_color(matrix[(i, j)]);
                    let line = graphics::Mesh::new_line(
                        ctx,
                        &[
                            glam::Vec2::new(
                                rect.x + rect.w * (i as f32 + 0.5) / WIDTH as f32,
                                rect.y
                                    + (starting_height as f32 - 1.0) * rect.h
                                        / (NUMBER_OF_LAYERS + 2) as f32,
                            ),
                            glam::Vec2::new(
                                rect.x + rect.w * (j as f32 + 0.5) / HEIGHT as f32,
                                rect.y
                                    + starting_height as f32 * rect.h
                                        / (NUMBER_OF_LAYERS + 2) as f32,
                            ),
                        ],
                        0.25,
                        graphics::Color::WHITE,
                    )?;
                    canvas.draw(&line, graphics::DrawParam::new().color(color));
                }
            }
        }

        return Ok(());
    }

    fn draw_network(
        canvas: &mut graphics::Canvas,
        brain: &Brain,
        ctx: &mut Context,
        rect: graphics::Rect,
        last_input_weights: &[f32; 10],
        last_output_weights: &[f32; 4],
    ) -> GameResult {
        let circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            mint::Point2 { x: 0.0, y: 0.0 },
            1.0,
            0.025,
            graphics::Color::WHITE,
        )?;

        for (i, &label) in INPUT_LABELS.iter().enumerate() {
            let mut text = graphics::Text::new(label);
            text.set_scale(16.0);
            text.set_layout(TextLayout {
                h_align: TextAlign::End,
                v_align: TextAlign::Middle,
            });
            canvas.draw(
                &text,
                graphics::DrawParam::default()
                    .scale(glam::Vec2::new(0.1, 0.1))
                    .rotation(std::f32::consts::FRAC_PI_2)
                    .dest(glam::Vec2::new(
                        rect.x + (i as f32 + 0.5) * rect.w / INPUT_LABELS.len() as f32,
                        rect.y + rect.h / (NUMBER_OF_LAYERS + 2) as f32 - 1.5,
                    )),
            );
        }

        for (i, &label) in OUTPUT_LABELS.iter().enumerate() {
            let mut text = graphics::Text::new(label);
            text.set_layout(TextLayout {
                h_align: TextAlign::Begin,
                v_align: TextAlign::Middle,
            });
            text.set_scale(16.0);
            canvas.draw(
                &text,
                graphics::DrawParam::default()
                    .scale(glam::Vec2::new(0.1, 0.1))
                    .rotation(std::f32::consts::FRAC_PI_2)
                    .dest(glam::Vec2::new(
                        rect.x + (i as f32 + 0.5) * rect.w / OUTPUT_LABELS.len() as f32,
                        rect.y
                            + rect.h * (NUMBER_OF_LAYERS) as f32 / (NUMBER_OF_LAYERS + 2) as f32
                            + 1.5,
                    )),
            );
        }

        for layer in 0..5 {
            let layer_size = if layer == 0 {
                10
            } else if layer == 4 {
                4
            } else {
                5
            };

            let y = rect.y + (layer + 1) as f32 * rect.h / (NUMBER_OF_LAYERS + 2) as f32;

            if layer == 0 {
            } else if layer == 1 {
                Self::draw_matrix(brain.get_inner().input_matrix, canvas, ctx, layer + 1, rect)?;
            } else if layer == 4 {
                Self::draw_matrix(
                    brain.get_inner().output_matrix,
                    canvas,
                    ctx,
                    layer + 1,
                    rect,
                )?;
            } else {
                Self::draw_matrix(
                    brain.get_inner().matricies[layer - 2].matrix,
                    canvas,
                    ctx,
                    layer + 1,
                    rect,
                )?;
            }

            for node in 0..layer_size {
                let color = if layer == 0 {
                    weight_to_color(last_input_weights[node])
                } else if layer == 4 {
                    weight_to_color(last_output_weights[node])
                } else {
                    Color::WHITE
                };

                canvas.draw(
                    &circle,
                    graphics::DrawParam::new()
                        .dest(glam::Vec2::new(
                            rect.x + (node as f32 + 0.5) * rect.w / layer_size as f32,
                            y,
                        ))
                        .color(color),
                )
            }
        }

        return Ok(());
    }

    fn create_random_wall_locations() -> Vec<glam::Vec2> {
        let mut rng = rand::rng();
        let mut wall_locations = Vec::new();

        let mut x = -ARENA_SIZE;
        while x < ARENA_SIZE {
            wall_locations.push(glam::Vec2::new(x, -ARENA_SIZE));
            wall_locations.push(glam::Vec2::new(x, ARENA_SIZE));
            if x != 0.0 && x != ARENA_SIZE * 2.0 {
                wall_locations.push(glam::Vec2::new(-ARENA_SIZE, x));
                wall_locations.push(glam::Vec2::new(ARENA_SIZE, x));
            }
            x += WALL_RADIUS * 2.0;
        }

        for _ in 0..NUMBER_OF_WALL_LOCATIONS - wall_locations.len() {
            let mut potential_location = glam::Vec2::new(
                rng.random_range(-ARENA_SIZE..ARENA_SIZE),
                rng.random_range(-ARENA_SIZE..ARENA_SIZE),
            );

            while potential_location.length_squared() < WALL_RADIUS * WALL_RADIUS * 4.0
                || wall_locations
                    .iter()
                    .any(|loc| loc.distance_squared(potential_location) < WALL_RADIUS * WALL_RADIUS)
            {
                potential_location = glam::Vec2::new(
                    rng.random_range(-ARENA_SIZE..ARENA_SIZE),
                    rng.random_range(-ARENA_SIZE..ARENA_SIZE),
                );
            }
            wall_locations.push(potential_location);
        }
        wall_locations
    }

    fn create_random_food_locations(wall_locations: &[glam::Vec2]) -> Vec<glam::Vec2> {
        let mut rng = rand::rng();
        let mut food_locations = Vec::new();
        for _ in 0..NUMBER_OF_FOOD_LOCATIONS {
            let mut potential_location = glam::Vec2::new(
                rng.random_range(-ARENA_SIZE..ARENA_SIZE),
                rng.random_range(-ARENA_SIZE..ARENA_SIZE),
            );
            while potential_location.length_squared() < FOOD_RADIUS * FOOD_RADIUS * 4.0
                || wall_locations.iter().any(|loc| {
                    (loc).distance_squared(potential_location)
                        < (WALL_RADIUS + FOOD_RADIUS) * (WALL_RADIUS + FOOD_RADIUS)
                })
                || food_locations.iter().any(|loc: &glam::Vec2| {
                    (loc).distance_squared(potential_location) < FOOD_RADIUS * FOOD_RADIUS
                })
            {
                potential_location = glam::Vec2::new(
                    rng.random_range(-ARENA_SIZE..ARENA_SIZE),
                    rng.random_range(-ARENA_SIZE..ARENA_SIZE),
                );
            }
            food_locations.push(potential_location);
        }
        food_locations
    }

    fn calculate_inputs(
        ship: &Spaceship,
        food_locations: &[glam::Vec2],
        wall_locations: &[glam::Vec2],
    ) -> [f32; 10] {
        let facing_direction = glam::Vec2::new(ship.angle.cos(), ship.angle.sin());

        let mut greens = 0.0;
        let mut greens_delta = 0.0;
        let mut horizontal_green_deta: f32 = 0.0;
        let mut reds = 0.0;
        let mut reds_delta = 0.0;

        let mut horizontal_red_delta = 0.0;

        for (index, food) in food_locations.iter().enumerate() {
            if ship.food[index] {
                let distance_squared = food.distance_squared(ship.location).max(1.0);

                greens += 1.0 / distance_squared;
                greens_delta +=
                    (*food - ship.location).normalize().dot(facing_direction) / distance_squared;
                horizontal_green_deta += (*food - ship.location)
                    .normalize()
                    .dot(facing_direction.perp())
                    / distance_squared;
            }
        }

        for wall in wall_locations {
            let distance_squared = wall.distance_squared(ship.location).max(1.0);

            reds += 1.0 / distance_squared;
            reds_delta +=
                (wall - ship.location).normalize().dot(facing_direction) / distance_squared;

            horizontal_red_delta += (wall - ship.location)
                .normalize()
                .dot(facing_direction.perp())
                / distance_squared;
        }

        return [
            greens * 16.,
            greens_delta * 16.,
            horizontal_green_deta * 16.,
            ship.velocity.dot(facing_direction),
            ship.velocity.dot(facing_direction.perp()),
            reds * 16.0,
            reds_delta * 16.0,
            horizontal_red_delta * 16.0,
            ship.angular_velocity,
            1.0,
        ];
    }
}

impl ggez::event::EventHandler for State {
    fn update(&mut self, _ctx: &mut ggez::Context) -> GameResult {
        self.steps += 1;

        let mut living_ships = 0;
        let mut weights_seen = false;

        for ship in self.spaceships.iter_mut() {
            if !ship.alive {
                continue;
            }

            let inputs = Self::calculate_inputs(ship, &self.food_locations, &self.wall_locations);

            if !weights_seen {
                self.last_input_weights = inputs
            }

            let forces = self.brains[ship.brain_index].get_inner().evaluate(inputs);

            if !weights_seen {
                self.last_output_weights = forces;
                weights_seen = true;
            }

            ship.angular_velocity *= 0.9;
            ship.velocity *= 0.99;

            for (force, (force_direction, position)) in forces.into_iter().zip([
                (glam::Vec2::new(1., 0.), glam::Vec2::new(0., 0.)),
                (glam::Vec2::new(-0.75, 0.), glam::Vec2::new(0., 0.)),
                (glam::Vec2::new(-0., -0.5), glam::Vec2::new(1., 0.)),
                (glam::Vec2::new(-0., 0.5), glam::Vec2::new(1., 0.)),
            ]) {
                ship.apply_force(force * force_direction, position);
            }

            ship.location += ship.velocity;
            ship.angle += ship.angular_velocity;

            if ship.time_last_food_eaten + 600 < self.steps {
                ship.die(self.steps);
                continue;
            }

            for (index, food) in self.food_locations.iter().enumerate() {
                if ship.food[index]
                    && ship.location.distance_squared(*food) <= FOOD_RADIUS * FOOD_RADIUS
                {
                    ship.eat(index, self.steps);
                    self.food_eaten[index] += 1;
                } else if ship.almost_food[index]
                    && ship.location.distance_squared(*food) <= FOOD_RADIUS * FOOD_RADIUS * 4.0
                {
                    ship.almost_food[index] = false;
                }
            }

            for wall in self.wall_locations.iter() {
                if ship.location.distance_squared(*wall) <= 10.0 {
                    ship.die(self.steps);
                    break;
                }
            }

            if ship.alive {
                living_ships += 1;
            }
        }

        if living_ships == 0 {
            self.round += 1;

            let individuals_with_fitness = std::mem::take(&mut self.brains)
                .into_iter()
                .enumerate()
                .map(|(i, brain)| {
                    let (count, sum) = self
                        .spaceships
                        .iter()
                        .filter(|k| k.brain_index == i)
                        .map(|ship| {
                            ship.food.iter().filter(|&i| !i).count() as f32 * 0.9375
                                + ship.almost_food.iter().filter(|&i| !i).count() as f32 * 0.0625
                        })
                        .fold((0.0, 0.0), |(count, sum), fitness| {
                            (count + 1.0, sum + fitness)
                        });
                    Individual {
                        brain: brain,
                        fitness: sum / count,
                    }
                })
                .collect::<Vec<_>>();

            self.best_fitness = individuals_with_fitness
                .iter()
                .map(|i| i.fitness)
                .reduce(|a, b| a.max(b))
                .unwrap_or(0.0);

            let mut rng = rand::rng();

            let population = evolution_rust::mutate(
                &mut rng,
                &mut self.persistant_info,
                individuals_with_fitness,
            );

            let save_file = std::fs::File::create("save.cbor")?;
            ciborium::into_writer(
                &(
                    self.round,
                    self.best_fitness,
                    &population,
                    &self.persistant_info,
                ),
                save_file,
            )
            .map_err(|err| error::GameError::CustomError(format!("{err:?}")))?;

            // Extract brains for evolution
            self.spaceships = (0..population.len())
                .map(create_ships_for_brain_index)
                .flatten()
                .collect();
            self.brains = population;

            // New Round, New Map
            self.wall_locations = State::create_random_wall_locations();
            self.food_locations = State::create_random_food_locations(&self.wall_locations);

            self.food_eaten.iter_mut().for_each(|i| *i = 0);

            self.steps = 0;
        }

        return Ok(());
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::BLACK);
        canvas.set_screen_coordinates(graphics::Rect::new(-50.0, -50.0, 150.0, 100.0));

        let rectangle = graphics::Mesh::new_polygon(
            ctx,
            graphics::DrawMode::fill(),
            &[
                glam::Vec2::new(-1.0, -0.8),
                glam::Vec2::new(1.0, -0.0),
                glam::Vec2::new(-1.0, 0.8),
            ],
            graphics::Color::WHITE,
        )?;

        for (index, ship) in self.spaceships.iter().enumerate() {
            canvas.draw(
                &rectangle,
                graphics::DrawParam::new()
                    .rotation(ship.angle)
                    .dest(ship.location)
                    .color(if index < 3 {
                        graphics::Color::BLUE
                    } else {
                        graphics::Color::WHITE
                    }),
            )
        }

        let food_circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            mint::Point2 { x: 0.0, y: 0.0 },
            FOOD_RADIUS,
            0.045,
            graphics::Color::WHITE,
        )?;

        for (index, &food) in self.food_locations.iter().enumerate() {
            let color = 1.0 - 4.0 * self.food_eaten[index] as f32 / self.spaceships.len() as f32;
            canvas.draw(
                &food_circle,
                graphics::DrawParam::new()
                    .dest(food)
                    .color(graphics::Color::new(
                        1.0 - color,
                        color * 4.0,
                        1.0 - 2.0 * color,
                        1.0,
                    )),
            );
        }

        let wall_circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            mint::Point2 { x: 0.0, y: 0.0 },
            WALL_RADIUS,
            0.1,
            graphics::Color::from_rgb(128, 12, 0),
        )?;

        for &wall in &self.wall_locations {
            canvas.draw(&wall_circle, graphics::DrawParam::new().dest(wall));
        }

        let mut text = graphics::Text::new(format!(
            "Round {}, Best Fitness {}, Alive: {}, fps: {:.1}",
            self.round,
            self.best_fitness,
            self.spaceships.iter().filter(|s| s.alive).count(),
            ctx.time.fps()
        ));
        text.set_scale(32.0);
        canvas.draw(
            &text,
            graphics::DrawParam::default()
                .scale(glam::Vec2::new(0.1, 0.1))
                .dest(glam::Vec2::new(-45.0, -45.0)),
        );

        Self::draw_network(
            &mut canvas,
            &self.brains[0],
            ctx,
            graphics::Rect::new(54.0, -48.0, 42.0, 96.0),
            &self.last_input_weights,
            &self.last_output_weights,
        )?;

        canvas.finish(ctx)?;
        Ok(())
    }
}

fn create_ships_for_brain_index(brain_index: usize) -> [Spaceship; 3] {
    [
        Spaceship::new(brain_index, 0.0),
        Spaceship::new(brain_index, PI * 2.0 / 3.0),
        Spaceship::new(brain_index, PI * 4.0 / 3.0),
    ]
}

fn main() -> Result<(), GameError> {
    let mut rng = rand::rng();

    let wall_locations = State::create_random_wall_locations();
    let food_locations = State::create_random_food_locations(&wall_locations);

    let state = if std::path::Path::new("save.cbor").exists() {
        let file = std::fs::File::open("save.cbor")?;
        let (round, best_fitness, population, persistant_info): (
            u32,
            f32,
            Vec<Brain>,
            evolution_rust::PersistentInfo,
        ) = ciborium::from_reader(file)
            .map_err(|err| GameError::CustomError(format!("{err:?}")))?;

        State {
            spaceships: (0..population.len())
                .map(create_ships_for_brain_index)
                .flatten()
                .collect(),
            round,
            best_fitness,
            steps: 0,
            food_eaten: vec![0; NUMBER_OF_FOOD_LOCATIONS],
            wall_locations,
            food_locations,
            persistant_info,
            brains: population,

            last_input_weights: [0.0; 10],
            last_output_weights: [0.0; 4],
        }
    } else {
        let (persistant_info, population): (evolution_rust::PersistentInfo, Vec<Brain>) =
            evolution_rust::initialize_random_population(100, 10, 10, &mut rng);

        State {
            spaceships: (0..population.len())
                .map(create_ships_for_brain_index)
                .flatten()
                .collect(),
            brains: population,
            steps: 0,
            round: 0,
            best_fitness: 0.0,
            food_eaten: vec![0; NUMBER_OF_FOOD_LOCATIONS],
            persistant_info,

            wall_locations,
            food_locations,

            last_input_weights: [0.0; 10],
            last_output_weights: [0.0; 4],
        }
    };
    let cb = ggez::ContextBuilder::new("rust_evolution", "mousetail")
        .window_setup(WindowSetup::default().title("Rust Evolution"))
        .window_mode(conf::WindowMode::default().dimensions(1536.0, 1024.0));
    let (ctx, event_loop) = cb.build().unwrap();
    event::run(ctx, event_loop, state)?;

    Ok(())
}
