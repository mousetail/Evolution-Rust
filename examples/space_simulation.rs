use ggez::{
    conf::WindowSetup,
    graphics::{TextAlign, TextLayout},
    *,
};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

type Brain = evolution_rust::Individual<10, 2, 4, 5>;

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
    brain: Brain,
    food: Vec<bool>,
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

    fn new(brain: Brain) -> Self {
        Spaceship {
            brain,
            food: vec![true; NUMBER_OF_FOOD_LOCATIONS],
            angle: 0.,
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
struct State {
    population: Vec<Spaceship>,
    steps: u32,
    round: u32,
    best_fitness: f32,
    food_eaten: Vec<u32>,

    wall_locations: Vec<glam::Vec2>,
    food_locations: Vec<glam::Vec2>,
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
                    let value = matrix[(i, j)].abs();
                    let value_u8 = ((value * 127.0) as u8);
                    let color = if matrix[(i, j)] > 0.0 {
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
                    };

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
                Self::draw_matrix(brain.input_matrix, canvas, ctx, layer + 1, rect)?;
            } else if layer == 4 {
                Self::draw_matrix(brain.output_matrix, canvas, ctx, layer + 1, rect)?;
            } else {
                Self::draw_matrix(brain.matricies[layer - 2], canvas, ctx, layer + 1, rect)?;
            }

            for node in 0..layer_size {
                canvas.draw(
                    &circle,
                    graphics::DrawParam::new().dest(glam::Vec2::new(
                        rect.x + (node as f32 + 0.5) * rect.w / layer_size as f32,
                        y,
                    )),
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

        for ship in self.population.iter_mut() {
            if !ship.alive {
                continue;
            }

            let forces = ship.brain.evaluate(Self::calculate_inputs(
                ship,
                &self.food_locations,
                &self.wall_locations,
            ));

            ship.angular_velocity *= 0.9;
            ship.velocity *= 0.99;

            for (force, (force_direction, position)) in forces.into_iter().zip([
                (glam::Vec2::new(1., 0.), glam::Vec2::new(0., 0.)),
                (glam::Vec2::new(-1., 0.), glam::Vec2::new(0., 0.)),
                (glam::Vec2::new(-0., -1.), glam::Vec2::new(1., 0.)),
                (glam::Vec2::new(-0., 1.), glam::Vec2::new(1., 0.)),
            ]) {
                ship.apply_force(force * force_direction, position);
            }

            ship.location += ship.velocity;
            ship.angle += ship.angular_velocity;

            if ship.time_last_food_eaten + 400 < self.steps {
                ship.die(self.steps);
                continue;
            }

            for (index, food) in self.food_locations.iter().enumerate() {
                if ship.food[index]
                    && ship.location.distance_squared(*food) <= FOOD_RADIUS * FOOD_RADIUS
                {
                    ship.eat(index, self.steps);
                    self.food_eaten[index] += 1;
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

            for ship in self.population.iter_mut() {
                ship.brain.fitness = ship.food.iter().filter(|&i| !i).count() as f32;
            }

            self.best_fitness = self
                .population
                .iter()
                .map(|i| i.brain.fitness)
                .reduce(|a, b| a.max(b))
                .unwrap_or(0.0);

            let save_file = std::fs::File::create("save.cbor")?;
            ciborium::into_writer(
                &(
                    self.round,
                    self.best_fitness,
                    self.population.iter().map(|i| &i.brain).collect::<Vec<_>>(),
                ),
                save_file,
            )
            .map_err(|err| error::GameError::CustomError(format!("{err:?}")))?;

            let mut population = evolution_rust::Population::new_from_individuals(
                10,
                self.population.iter().map(|i| i.brain.clone()).collect(),
            );
            population.evolve(&mut rand::rng());

            self.population = population
                .individuals
                .into_iter()
                .map(Spaceship::new)
                .collect();
            let new_random_angle = rand::rng().random_range(-3.0..3.0);
            self.population
                .iter_mut()
                .for_each(|i| i.angle = new_random_angle);

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

        for (index, ship) in self.population.iter().enumerate() {
            canvas.draw(
                &rectangle,
                graphics::DrawParam::new()
                    .rotation(ship.angle)
                    .dest(ship.location)
                    .color(if index == 0 {
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
            1.0,
            0.025,
            graphics::Color::WHITE,
        )?;

        for (index, &food) in self.food_locations.iter().enumerate() {
            let color = 1.0 - 4.0 * self.food_eaten[index] as f32 / self.population.len() as f32;
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
            self.population.iter().filter(|s| s.alive).count(),
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
            &self.population[0].brain,
            ctx,
            graphics::Rect::new(54.0, -48.0, 42.0, 96.0),
        )?;

        canvas.finish(ctx)?;
        Ok(())
    }
}

fn main() -> Result<(), GameError> {
    let mut rng = rand::rng();
    let population = evolution_rust::Population::new(100, 10, &mut rng);

    let wall_locations = State::create_random_wall_locations();
    let food_locations = State::create_random_food_locations(&wall_locations);

    let state = if std::path::Path::new("save.cbor").exists() {
        let file = std::fs::File::open("save.cbor")?;
        let (round, best_fitness, population): (u32, f32, Vec<Brain>) = ciborium::from_reader(file)
            .map_err(|err| GameError::CustomError(format!("{err:?}")))?;

        State {
            population: population.into_iter().map(Spaceship::new).collect(),
            round,
            best_fitness,
            steps: 0,
            food_eaten: vec![0; NUMBER_OF_FOOD_LOCATIONS],
            wall_locations,
            food_locations,
        }
    } else {
        State {
            population: population
                .individuals
                .into_iter()
                .map(Spaceship::new)
                .collect(),
            steps: 0,
            round: 0,
            best_fitness: 0.0,
            food_eaten: vec![0; NUMBER_OF_FOOD_LOCATIONS],

            wall_locations,
            food_locations,
        }
    };
    let cb = ggez::ContextBuilder::new("rust_evolution", "mousetail")
        .window_setup(WindowSetup::default().title("Rust Evolution"))
        .window_mode(conf::WindowMode::default().dimensions(1536.0, 1024.0));
    let (ctx, event_loop) = cb.build().unwrap();
    event::run(ctx, event_loop, state)?;

    Ok(())
}
