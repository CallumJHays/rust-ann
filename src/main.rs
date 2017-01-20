extern crate pretty_env_logger;
#[macro_use]
extern crate log;
extern crate rand;

mod network;
mod activators;

use network::{Network, LayerBlueprint};
use rand::Rng;
pub use activators::{Activator, Sigmoid, Elu};

fn main() {
    pretty_env_logger::init().unwrap();


    // uniformly random seed weights for the output layer
    let output_layer_seed_weights = {
        let mut rng = rand::thread_rng();
        vec![(0..10).map(|_| rng.gen()).collect()]
    };

    // create a network with 3 layers:
    // the first having three nodes using Sigmoid activation and an initial bias of 0.5
    // the next having 10 nodes using Elu activation
    // the final (output) layer having 1 node and a pre-defined seed weight
    let mut nn = Network::new(3,
                              vec![LayerBlueprint::new(4).bias(0.5).activator(Sigmoid),
                                   LayerBlueprint::new(10).activator(Elu),
                                   LayerBlueprint::new(1).weights(output_layer_seed_weights)]);

    // when `None` is specified, the most common defaults are used
    // IE xavier initialization for seed weights, uniform random for bias, Relu for activation.
    // This network is not suited for the task, but is just an example

    // define a simple test function, output = input[0];
    let inputs = vec![vec![0., 0., 1.], vec![0., 1., 1.], vec![1., 0., 1.], vec![1., 1., 1.]];
    let outputs = vec![0., 0., 1., 1.];

    // train 10'000 times with a variable learning rate
    let mut learning_rate = 0.01;

    for i in 0..10000 {
        let mut errors = vec![];
        for input in &inputs {
            for output in &outputs {
                errors.push(nn.train(input, &[*output], learning_rate));
            }
        }

        let mean_errors = errors.iter().sum::<f64>() / (errors.len() as f64);

        if i % 100 == 0 {
            println!("{}: {}", i, mean_errors);
            learning_rate *= 0.9; // slow down learning every 100 steps
        }
    }
}
