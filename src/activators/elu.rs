use super::Activator;

#[derive(Debug)]
pub struct Elu;

impl Activator for Elu {
    fn activate(&self, x: f64) -> f64 {
        if x < 0. { x.exp() - 1. } else { x }
    }

    fn derived(&self, x: f64) -> f64 {
        if x < 0. { x.exp() } else { 1. }
    }
}
