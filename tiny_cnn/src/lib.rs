pub mod layers;
pub mod model;
pub mod trainer;
pub mod inference;
pub mod persistence;

pub use model::Model;
pub use trainer::Trainer;
pub use inference::Inference;
pub use persistence::ModelPersistence;

pub mod prelude {
    pub use crate::Model;
    pub use crate::Trainer;
    pub use crate::Inference;
    pub use crate::ModelPersistence;
} 