pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/lost_cities.v1.rs"));
}

mod config;
mod engine;
mod error;
mod state;

pub use config::Config;
pub use engine::LostCitiesEngine;
pub use error::{EngineError, EngineErrorKind};
pub use state::{score_expedition, Card, GameState, Phase};
