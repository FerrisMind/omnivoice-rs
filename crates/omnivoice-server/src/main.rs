use clap::Parser;
use omnivoice_server::{
    build_router,
    error::ServerError,
    runtime::{AppState, PipelineSpeechRuntime, ServerConfig},
    ServerArgs,
};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), ServerError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "omnivoice_server=info,tower_http=info".into()),
        )
        .with_target(false)
        .init();

    let args = ServerArgs::parse();
    let runtime_options = args.runtime_options()?;
    let runtime = PipelineSpeechRuntime::from_options(runtime_options)?;
    let config = ServerConfig::from_args(&args)?;
    let host = args.host.clone();
    let port = args.port;

    let app = build_router(AppState::new(runtime, config));
    let listener = TcpListener::bind((host.as_str(), port)).await?;

    info!("omnivoice-server listening on http://{host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
