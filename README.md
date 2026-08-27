# Evolution-Rust

To run the example:

```
cargo run --example space_simulation
````

# How it Works

This is my own custom method of training neural networks. It's heavily inspired by NEAT but is also completely different because I just use plain N by N matricies for performance rather than how NEAT can create arbitrairy non-rectangular topologies.

Basically, each individual is part of a species, and it tries to keep the best of each species alive for a bit of time so they have time to reach their local maximum. Every few generations, we do a mass extinction events and kill of underpeforming species and split the remaining ones into new species. This reduces the chance we get stuck in a local optima where small variantions hurt fitness but a better strategy is further away.

We also try to randomly either peform smaller mutations or larger ones to balance between small optimiations for fine tuning and exploring entirely new state spaces.
