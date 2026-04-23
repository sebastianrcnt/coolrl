use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use crate::error::{EngineError, EngineErrorKind};
use crate::proto;
use crate::LostCitiesEngine;

#[derive(Clone, Default)]
pub struct LostCitiesGrpcService {
    engine: Arc<Mutex<LostCitiesEngine>>,
}

impl LostCitiesGrpcService {
    pub fn new(engine: LostCitiesEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
        }
    }

    fn with_engine<T>(
        &self,
        f: impl FnOnce(&mut LostCitiesEngine) -> Result<T, EngineError>,
    ) -> Result<T, Status> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| Status::internal("lost cities engine mutex poisoned"))?;
        f(&mut engine).map_err(map_engine_error)
    }
}

#[tonic::async_trait]
impl proto::lost_cities_server::LostCities for LostCitiesGrpcService {
    async fn new_game(
        &self,
        request: Request<proto::NewGameRequest>,
    ) -> Result<Response<proto::GameObservation>, Status> {
        let observation = self.with_engine(|engine| engine.new_game(request.into_inner()))?;
        Ok(Response::new(observation))
    }

    async fn get_observation(
        &self,
        request: Request<proto::SessionRef>,
    ) -> Result<Response<proto::GameObservation>, Status> {
        let observation =
            self.with_engine(|engine| engine.get_observation(request.into_inner()))?;
        Ok(Response::new(observation))
    }

    async fn apply_action(
        &self,
        request: Request<proto::ApplyActionRequest>,
    ) -> Result<Response<proto::StepResult>, Status> {
        let result = self.with_engine(|engine| engine.apply_action(request.into_inner()))?;
        Ok(Response::new(result))
    }

    async fn end_session(
        &self,
        request: Request<proto::SessionRef>,
    ) -> Result<Response<()>, Status> {
        self.with_engine(|engine| engine.end_session(request.into_inner()))?;
        Ok(Response::new(()))
    }
}

fn map_engine_error(error: EngineError) -> Status {
    match error.kind() {
        EngineErrorKind::AlreadyExists => Status::already_exists(error.message().to_string()),
        EngineErrorKind::NotFound => Status::not_found(error.message().to_string()),
        EngineErrorKind::FailedPrecondition => {
            Status::failed_precondition(error.message().to_string())
        }
        EngineErrorKind::InvalidArgument => Status::invalid_argument(error.message().to_string()),
    }
}
