use lost_cities_core::proto::{
    self, lost_cities_client::LostCitiesClient, lost_cities_server::LostCitiesServer,
};
use lost_cities_core::LostCitiesGrpcService;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Endpoint, Server};
use tonic::Code;

fn small_config(seed: u64) -> proto::GameConfig {
    proto::GameConfig {
        n_colors: 2,
        n_ranks: 2,
        min_rank: 1,
        n_handshakes: 0,
        hand_size: 1,
        expedition_penalty: 0,
        bonus_threshold: 99,
        bonus_amount: 0,
        seed: Some(seed),
    }
}

async fn spawn_client() -> (LostCitiesClient<Channel>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener should bind");
    let addr = listener.local_addr().expect("listener addr");
    let incoming = TcpListenerStream::new(listener);

    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(LostCitiesServer::new(LostCitiesGrpcService::default()))
            .serve_with_incoming(incoming)
            .await
            .expect("gRPC server should run");
    });

    let endpoint = Endpoint::from_shared(format!("http://{}", addr)).expect("endpoint");
    let client = LostCitiesClient::new(
        endpoint
            .connect()
            .await
            .expect("client should connect to server"),
    );
    (client, server)
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_round_trip_new_game_and_get_observation() {
    let (mut client, server) = spawn_client().await;

    let observation = client
        .new_game(proto::NewGameRequest {
            session_id: "grpc-round-trip".to_string(),
            config: Some(small_config(13)),
        })
        .await
        .expect("new_game should succeed")
        .into_inner();
    assert_eq!(observation.state_version, 0);
    assert_eq!(observation.current_player, 0);
    assert_eq!(observation.phase, proto::Phase::Card as i32);

    let opponent_view = client
        .get_observation(proto::SessionRef {
            session_id: "grpc-round-trip".to_string(),
            observer_player: Some(1),
        })
        .await
        .expect("get_observation should succeed")
        .into_inner();
    assert_eq!(opponent_view.observer_player, 1);
    assert!(opponent_view
        .legal_actions
        .expect("legal actions should exist")
        .actions
        .is_empty());

    server.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn grpc_maps_failed_precondition_for_stale_version() {
    let (mut client, server) = spawn_client().await;

    let observation = client
        .new_game(proto::NewGameRequest {
            session_id: "grpc-stale".to_string(),
            config: Some(small_config(21)),
        })
        .await
        .expect("new_game should succeed")
        .into_inner();
    let action_id = observation
        .legal_actions
        .as_ref()
        .and_then(|actions| actions.actions.first())
        .map(|action| action.id)
        .expect("must have a legal action");

    client
        .apply_action(proto::ApplyActionRequest {
            session_id: "grpc-stale".to_string(),
            action_id,
            expected_state_version: observation.state_version,
            observer_player: None,
        })
        .await
        .expect("first action should succeed");

    let error = client
        .apply_action(proto::ApplyActionRequest {
            session_id: "grpc-stale".to_string(),
            action_id,
            expected_state_version: observation.state_version,
            observer_player: None,
        })
        .await
        .expect_err("stale state_version must fail");
    assert_eq!(error.code(), Code::FailedPrecondition);

    server.abort();
}
