use ggez::{conf::WindowSetup, *};
use rand::Rng;
use serde::{Deserialize, Serialize};

type Brain = evolution_rust::Individual<8, 2, 6, 4>;

#[derive(Serialize, Deserialize)]
struct Spaceship {
    brain: Brain,
    food: Vec<bool>,
    location: glam::Vec2,
    angle: f32,
    alive: bool,
    velocity: glam::Vec2,
    angular_velocity: f32,
}

impl Spaceship {
    const MASS: f32 = 4.0;
    const MOMENT_OF_INERTIA: f32 = 16.0;

    fn new(brain: Brain) -> Self {
        Spaceship {
            brain,
            food: FOOD_LOCATIONS.iter().map(|_| true).collect(),
            angle: 0.,
            location: glam::Vec2::new(0., 0.),
            alive: true,
            velocity: glam::Vec2::new(0., 0.),
            angular_velocity: 0.,
        }
    }

    fn apply_force(&mut self, force: glam::Vec2, position: glam::Vec2) {
        self.angular_velocity +=
            glam::Vec2::new(position.y, position.x).dot(force) / Self::MOMENT_OF_INERTIA;
        self.velocity += force.rotate(glam::Vec2::from_angle(self.angle)) / Self::MASS;
    }
}
struct State {
    population: Vec<Spaceship>,
    steps: u32,
    round: u32,
    best_fitness: f32,
    food_eaten: Vec<u32>,
}

impl ggez::event::EventHandler<GameError> for State {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        self.steps += 1;

        let mut living_ships = 0;

        for ship in self.population.iter_mut() {
            if !ship.alive {
                continue;
            }

            let facing_direction = glam::Vec2::new(ship.angle.cos(), ship.angle.sin());

            let mut greens = 0.0;
            let mut greens_delta = 0.0;
            let mut horizontal_green_deta: f32 = 0.0;
            let mut reds = 0.0;
            let mut reds_delta = 0.0;

            let mut horizontal_red_delta = 0.0;

            for (index, food) in FOOD_LOCATIONS.iter().enumerate() {
                if ship.food[index] {
                    let distance_squared = food.distance_squared(ship.location).max(1.0);

                    greens += 1.0 / distance_squared;
                    greens_delta += (*food - ship.location).normalize().dot(facing_direction)
                        / distance_squared;
                    horizontal_green_deta += (*food - ship.location)
                        .normalize()
                        .dot(facing_direction.perp())
                        / distance_squared;
                }
            }

            for wall in WALL_LOCATIONS {
                let distance_squared = wall.distance_squared(ship.location).max(1.0);

                reds += 1.0 / distance_squared;
                reds_delta +=
                    (wall - ship.location).normalize().dot(facing_direction) / distance_squared;

                horizontal_red_delta += (wall - ship.location)
                    .normalize()
                    .dot(facing_direction.perp())
                    / distance_squared;
            }

            let forces = ship.brain.evaluate([
                greens * 16.,
                greens_delta * 16.,
                horizontal_green_deta * 16.,
                reds * 16.0,
                reds_delta * 16.0,
                horizontal_red_delta * 16.0,
                ship.angular_velocity,
                1.0,
            ]);

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

            for (index, food) in FOOD_LOCATIONS.iter().enumerate() {
                if ship.food[index] && ship.location.distance_squared(*food) <= 2.0 {
                    ship.food[index] = false;
                    ship.brain.fitness += 1.0;
                    self.food_eaten[index] += 1;
                }
            }

            for wall in WALL_LOCATIONS.iter() {
                if ship.location.distance_squared(*wall) <= 10.0 {
                    ship.alive = false;
                    ship.brain.fitness += self.steps as f32 / 1000.0;
                    break;
                }
            }

            if ship.alive {
                living_ships += 1;
            }
        }

        if living_ships == 0 || self.steps >= if self.round == 0 { 1000 } else { 2000 } {
            self.round += 1;
            self.best_fitness = self
                .population
                .iter()
                .map(|i| i.brain.fitness)
                .reduce(|a, b| a.max(b))
                .unwrap_or(0.0);

            let save_file = std::fs::File::create("save.cbor")?;
            ciborium::into_writer(
                &(self.round, self.best_fitness, &self.population),
                save_file,
            )
            .map_err(|err| error::GameError::CustomError(format!("{err:?}")))?;

            let mut population = evolution_rust::Population::new_from_individuals(
                10,
                self.population.iter().map(|i| i.brain.clone()).collect(),
            );
            population.evolve(&mut rand::thread_rng());

            self.population = population
                .individuals
                .into_iter()
                .map(Spaceship::new)
                .collect();
            let new_random_angle = rand::thread_rng().gen_range(-3.0..3.0);
            self.population
                .iter_mut()
                .for_each(|i| i.angle = new_random_angle);

            self.food_eaten.iter_mut().for_each(|i| *i = 0);

            self.steps = 0;
        }

        return Ok(());
    }
    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::BLACK);
        canvas.set_screen_coordinates(graphics::Rect::new(-50.0, -50.0, 100.0, 100.0));

        let rectangle = graphics::Mesh::new_rectangle(
            ctx,
            graphics::DrawMode::fill(),
            graphics::Rect::new(-0.8, -1.0, 2.0, 1.6),
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
            graphics::Color::GREEN,
        )?;

        for (index, food) in FOOD_LOCATIONS.into_iter().enumerate() {
            let color = 1.0 - self.food_eaten[index] as f32 / self.population.len() as f32;
            canvas.draw(
                &food_circle,
                graphics::DrawParam::new()
                    .dest(food)
                    .color(graphics::Color::new(color, color, color, 1.0)),
            );
        }

        let wall_circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            mint::Point2 { x: 0.0, y: 0.0 },
            3.0,
            0.1,
            graphics::Color::RED,
        )?;

        for wall in WALL_LOCATIONS {
            canvas.draw(&wall_circle, graphics::DrawParam::new().dest(wall));
        }

        let mut text = graphics::Text::new(format!(
            "Round {}, Best Fitness {}, fps: {:.1}",
            self.round,
            self.best_fitness,
            ctx.time.fps()
        ));
        text.set_scale(32.0);
        canvas.draw(
            &text,
            graphics::DrawParam::default()
                .scale(glam::Vec2::new(0.1, 0.1))
                .dest(glam::Vec2::new(-45.0, -45.0)),
        );

        canvas.finish(ctx)?;
        Ok(())
    }
}

static FOOD_LOCATIONS: [glam::Vec2; 118] = [
    glam::Vec2::new(0., -12.),
    glam::Vec2::new(6., -25.),
    glam::Vec2::new(12., 41.),
    glam::Vec2::new(19., 8.),
    glam::Vec2::new(0., -25.),
    glam::Vec2::new(6., -8.),
    glam::Vec2::new(12., 14.),
    glam::Vec2::new(-12., 8.),
    glam::Vec2::new(-7.620772941740668, -42.794800208328525),
    glam::Vec2::new(-46.82143214531385, -20.055170686389694),
    glam::Vec2::new(16.106658915371156, -43.48280725415813),
    glam::Vec2::new(-34.73769600107101, 23.324621318512367),
    glam::Vec2::new(-11.566904152330103, 4.389390173107939),
    glam::Vec2::new(-27.992831581525664, -44.638735559292286),
    glam::Vec2::new(7.765738257574078, 6.512589854707151),
    glam::Vec2::new(30.144901815020685, 35.58778590424604),
    glam::Vec2::new(39.771696787117385, 20.125712105176987),
    glam::Vec2::new(-1.818778089329638, 28.360784358666958),
    glam::Vec2::new(14.907165255080175, 31.885570656752865),
    glam::Vec2::new(32.251495463767576, -24.148212298992256),
    glam::Vec2::new(28.645297908559133, -2.652983499525008),
    glam::Vec2::new(-37.66124868498243, -28.722387341531395),
    glam::Vec2::new(-9.178449911781474, -30.204443412254967),
    glam::Vec2::new(-38.08968966862823, 17.778114230926498),
    glam::Vec2::new(35.82867775449527, -15.328261864822856),
    glam::Vec2::new(32.966640725744526, 18.975950566940423),
    glam::Vec2::new(31.126535830112587, 36.66039829072854),
    glam::Vec2::new(11.8268698636903, 12.006221625181322),
    glam::Vec2::new(-33.08192904716574, -19.611015142612573),
    glam::Vec2::new(-46.409618977466195, -9.512491045103399),
    glam::Vec2::new(-36.063910460930224, -28.639274022373453),
    glam::Vec2::new(-20.932181878157667, -31.97654524652848),
    glam::Vec2::new(-28.72479058333578, 42.9054799775039),
    glam::Vec2::new(-23.828102675327166, -11.088258035243882),
    glam::Vec2::new(-12.573055213680837, 21.471363508268126),
    glam::Vec2::new(-9.600609778544293, -36.96664806839588),
    glam::Vec2::new(-21.318484683473322, -17.97136175448355),
    glam::Vec2::new(-27.016372873357636, 36.70098316566665),
    glam::Vec2::new(2.0763823777282964, -43.92509106747539),
    glam::Vec2::new(-25.828702562194973, 45.064523168432224),
    glam::Vec2::new(23.87066286427663, 18.976149530940106),
    glam::Vec2::new(35.433014850884206, -34.19502156086414),
    glam::Vec2::new(-23.371271147614213, -31.155903956340918),
    glam::Vec2::new(17.381198810479162, -37.63663639258773),
    glam::Vec2::new(-20.104824245559087, 46.49261837552774),
    glam::Vec2::new(42.959611286733384, -36.93751892756958),
    glam::Vec2::new(-25.328968380852533, -34.91476806869031),
    glam::Vec2::new(-43.64745826715408, -40.051200791586844),
    glam::Vec2::new(-16.941936397175215, 29.0779892092548),
    glam::Vec2::new(-21.7219030467798, 24.406808384885903),
    glam::Vec2::new(-32.815359597282345, 34.670994493719675),
    glam::Vec2::new(-3.876517710390012, -16.410907691243022),
    glam::Vec2::new(31.489825861031854, -17.363199857459623),
    glam::Vec2::new(-46.34758160288273, -10.265540512916187),
    glam::Vec2::new(-29.63125182366285, -0.05414939846848599),
    glam::Vec2::new(-6.222534867281012, -25.6047090028603),
    glam::Vec2::new(18.880504722554456, 6.7694347908520465),
    glam::Vec2::new(20.256418892358365, -31.512797375890322),
    glam::Vec2::new(30.496205733916888, -23.879467930271485),
    glam::Vec2::new(15.238272969232316, -31.51003572791919),
    glam::Vec2::new(28.44568524182792, -10.585100771303356),
    glam::Vec2::new(-36.936317666352224, 37.46946090025241),
    glam::Vec2::new(-2.8836162438376007, 12.595559260829749),
    glam::Vec2::new(0.39824696190816566, -12.13433227573875),
    glam::Vec2::new(-25.56977075034023, 19.98151616465212),
    glam::Vec2::new(-18.017859933244495, 4.034220673148447),
    glam::Vec2::new(-29.521666797320258, -27.2667553843009),
    glam::Vec2::new(-14.552449425706698, -43.258010754490435),
    glam::Vec2::new(5.084092766446378, -39.21829284876206),
    glam::Vec2::new(19.777504824445202, -44.25394722869699),
    glam::Vec2::new(11.536986981918556, 8.82227726865802),
    glam::Vec2::new(31.037971151461726, -11.291394310703776),
    glam::Vec2::new(2.688951614438075, -25.05450613003052),
    glam::Vec2::new(33.192925216701866, 11.541928396006416),
    glam::Vec2::new(-5.685521386403871, 25.682304172147703),
    glam::Vec2::new(21.379300732674487, 25.94867981895117),
    glam::Vec2::new(26.12340319929507, 29.320121721104137),
    glam::Vec2::new(-3.9028085496422227, -45.91296995873273),
    glam::Vec2::new(-5.878794291875636, 7.617025557220211),
    glam::Vec2::new(12.056847106398317, 40.64585414508065),
    glam::Vec2::new(-29.032883355196084, -26.367379318811075),
    glam::Vec2::new(46.60869990559083, -12.660621341896958),
    glam::Vec2::new(43.265517842679664, -22.962785328368593),
    glam::Vec2::new(46.51457757850717, 46.031264747360424),
    glam::Vec2::new(6.032901734172901, 42.729102383384316),
    glam::Vec2::new(-37.381581510329724, -40.50230423951367),
    glam::Vec2::new(25.821243391482295, 26.428933143553444),
    glam::Vec2::new(12.743962409067311, -24.167395342859752),
    glam::Vec2::new(-10.402744390399079, 26.601629773526607),
    glam::Vec2::new(24.154065229496947, 41.62789425134857),
    glam::Vec2::new(42.142975056213714, 42.228564706193666),
    glam::Vec2::new(-14.043709104944865, 40.576258994678085),
    glam::Vec2::new(45.363919951112216, -0.7224549284427455),
    glam::Vec2::new(-38.120494809646424, 23.418418771128504),
    glam::Vec2::new(8.311415975686621, -1.198489516042726),
    glam::Vec2::new(-40.86456200626991, 21.58234361058239),
    glam::Vec2::new(20.995261565338335, -21.821789621403298),
    glam::Vec2::new(-39.19176883806685, 29.370891799025348),
    glam::Vec2::new(16.474187521055256, 34.15800102099248),
    glam::Vec2::new(1.3271866658165334, -37.6277829172805),
    glam::Vec2::new(22.73482851431915, 24.007795464529703),
    glam::Vec2::new(4.170946582722074, 30.53692188003356),
    glam::Vec2::new(-40.96998573099675, -44.58658122909628),
    glam::Vec2::new(-15.71273933697701, 21.362021710770268),
    glam::Vec2::new(38.318509368868966, 11.233931245517267),
    glam::Vec2::new(14.133730967803727, 41.31092617501651),
    glam::Vec2::new(46.069090700733206, 37.75732640494331),
    glam::Vec2::new(45.238028439011366, 25.641827671328514),
    glam::Vec2::new(-0.9078255659594292, 46.88065469742662),
    glam::Vec2::new(-9.996864225727876, -15.800001028805848),
    glam::Vec2::new(29.63567795082347, 35.582688641564665),
    glam::Vec2::new(15.330799643678196, 27.372136208332957),
    glam::Vec2::new(-33.49374925001522, -40.861345049371415),
    glam::Vec2::new(31.91812584517793, -29.563316179967916),
    glam::Vec2::new(-5.894153287120611, -11.080854129087566),
    glam::Vec2::new(-25.16602168006189, 29.67217016997148),
    glam::Vec2::new(-34.0539210629473, -40.6566312298508),
    glam::Vec2::new(35.36240234245935, -30.68441504351222),
];
static WALL_LOCATIONS: [glam::Vec2; 100] = [
    glam::Vec2::new(-43.24182006985064, 37.02130316244868),
    glam::Vec2::new(-19.57081002728108, 0.5046357786797826),
    glam::Vec2::new(38.23447366331598, -38.86666787947049),
    glam::Vec2::new(-6.711219430147191, -44.255892341989785),
    glam::Vec2::new(-12.365574053587586, -6.700611294541883),
    glam::Vec2::new(-15.673827451480442, -1.312038797658731),
    glam::Vec2::new(-7.703075369774032, -41.43800538507749),
    glam::Vec2::new(-17.610129138274214, -16.112118311555772),
    glam::Vec2::new(37.22413428196374, -40.413874697570904),
    glam::Vec2::new(9.850116784938079, 37.96665074797123),
    glam::Vec2::new(-5.471904888307466, 15.036286922423168),
    glam::Vec2::new(-39.776400434096765, -45.532860432349175),
    glam::Vec2::new(10.85039826102731, 12.596318056091),
    glam::Vec2::new(44.83268497722997, -10.441228071229965),
    glam::Vec2::new(10.989823452129254, -26.44453593532225),
    glam::Vec2::new(16.21868031772715, -29.40553651312599),
    glam::Vec2::new(7.686746149559425, -12.264007481497531),
    glam::Vec2::new(-31.754434594077516, 17.089489861049504),
    glam::Vec2::new(-43.87963021604549, 20.698098854164876),
    glam::Vec2::new(-12.964413577233216, -10.399441253553327),
    glam::Vec2::new(-50.0, -50.0),
    glam::Vec2::new(-50.0, -50.0),
    glam::Vec2::new(50.0, -50.0),
    glam::Vec2::new(-50.0, 50.0),
    glam::Vec2::new(-50.0, -45.0),
    glam::Vec2::new(-45.0, -50.0),
    glam::Vec2::new(50.0, -45.0),
    glam::Vec2::new(-45.0, 50.0),
    glam::Vec2::new(-50.0, -40.0),
    glam::Vec2::new(-40.0, -50.0),
    glam::Vec2::new(50.0, -40.0),
    glam::Vec2::new(-40.0, 50.0),
    glam::Vec2::new(-50.0, -35.0),
    glam::Vec2::new(-35.0, -50.0),
    glam::Vec2::new(50.0, -35.0),
    glam::Vec2::new(-35.0, 50.0),
    glam::Vec2::new(-50.0, -30.0),
    glam::Vec2::new(-30.0, -50.0),
    glam::Vec2::new(50.0, -30.0),
    glam::Vec2::new(-30.0, 50.0),
    glam::Vec2::new(-50.0, -25.0),
    glam::Vec2::new(-25.0, -50.0),
    glam::Vec2::new(50.0, -25.0),
    glam::Vec2::new(-25.0, 50.0),
    glam::Vec2::new(-50.0, -20.0),
    glam::Vec2::new(-20.0, -50.0),
    glam::Vec2::new(50.0, -20.0),
    glam::Vec2::new(-20.0, 50.0),
    glam::Vec2::new(-50.0, -15.0),
    glam::Vec2::new(-15.0, -50.0),
    glam::Vec2::new(50.0, -15.0),
    glam::Vec2::new(-15.0, 50.0),
    glam::Vec2::new(-50.0, -10.0),
    glam::Vec2::new(-10.0, -50.0),
    glam::Vec2::new(50.0, -10.0),
    glam::Vec2::new(-10.0, 50.0),
    glam::Vec2::new(-50.0, -5.0),
    glam::Vec2::new(-5.0, -50.0),
    glam::Vec2::new(50.0, -5.0),
    glam::Vec2::new(-5.0, 50.0),
    glam::Vec2::new(-50.0, 0.0),
    glam::Vec2::new(0.0, -50.0),
    glam::Vec2::new(50.0, 0.0),
    glam::Vec2::new(0.0, 50.0),
    glam::Vec2::new(-50.0, 5.0),
    glam::Vec2::new(5.0, -50.0),
    glam::Vec2::new(50.0, 5.0),
    glam::Vec2::new(5.0, 50.0),
    glam::Vec2::new(-50.0, 10.0),
    glam::Vec2::new(10.0, -50.0),
    glam::Vec2::new(50.0, 10.0),
    glam::Vec2::new(10.0, 50.0),
    glam::Vec2::new(-50.0, 15.0),
    glam::Vec2::new(15.0, -50.0),
    glam::Vec2::new(50.0, 15.0),
    glam::Vec2::new(15.0, 50.0),
    glam::Vec2::new(-50.0, 20.0),
    glam::Vec2::new(20.0, -50.0),
    glam::Vec2::new(50.0, 20.0),
    glam::Vec2::new(20.0, 50.0),
    glam::Vec2::new(-50.0, 25.0),
    glam::Vec2::new(25.0, -50.0),
    glam::Vec2::new(50.0, 25.0),
    glam::Vec2::new(25.0, 50.0),
    glam::Vec2::new(-50.0, 30.0),
    glam::Vec2::new(30.0, -50.0),
    glam::Vec2::new(50.0, 30.0),
    glam::Vec2::new(30.0, 50.0),
    glam::Vec2::new(-50.0, 35.0),
    glam::Vec2::new(35.0, -50.0),
    glam::Vec2::new(50.0, 35.0),
    glam::Vec2::new(35.0, 50.0),
    glam::Vec2::new(-50.0, 40.0),
    glam::Vec2::new(40.0, -50.0),
    glam::Vec2::new(50.0, 40.0),
    glam::Vec2::new(40.0, 50.0),
    glam::Vec2::new(-50.0, 45.0),
    glam::Vec2::new(45.0, -50.0),
    glam::Vec2::new(50.0, 45.0),
    glam::Vec2::new(45.0, 50.0),
];

fn main() -> Result<(), GameError> {
    let mut rng = rand::thread_rng();
    let population = evolution_rust::Population::new(100, 10, &mut rng);

    let state = if std::path::Path::new("save.cbor").exists() {
        let file = std::fs::File::open("save.cbor")?;
        let (round, best_fitness, population): (u32, f32, Vec<Spaceship>) =
            ciborium::from_reader(file)
                .map_err(|err| GameError::CustomError(format!("{err:?}")))?;

        State {
            population: population
                .iter()
                .map(|i| Spaceship::new(i.brain.clone()))
                .collect(),
            round,
            best_fitness,
            steps: 0,
            food_eaten: FOOD_LOCATIONS.iter().map(|_| 0).collect(),
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
            food_eaten: FOOD_LOCATIONS.iter().map(|_| 0).collect(),
        }
    };
    let cb = ggez::ContextBuilder::new("rust_evolution", "mousetail")
        .window_setup(WindowSetup::default().title("Rust Evolution"))
        .window_mode(conf::WindowMode::default().dimensions(1024.0, 1024.0));
    let (ctx, event_loop) = cb.build().unwrap();
    event::run(ctx, event_loop, state);

    Ok(())
}
