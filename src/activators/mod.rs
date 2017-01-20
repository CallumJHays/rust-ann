use std::fmt::Debug;

mod relu;
pub use self::relu::Relu;

mod elu;
pub use self::elu::Elu;

mod sigmoid;
pub use self::sigmoid::Sigmoid;

// different activators can be used to train neural networks.
// They all share the same API so they can be defined as a trait!
pub trait Activator: Debug {
    fn activate(&self, f64) -> f64;

    fn derived(&self, f64) -> f64;
}
