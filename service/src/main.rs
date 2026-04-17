use axum::{routing::{get, post}, Json, Router, extract::State};
use ort::{GraphOptimizationLevel, Session, SessionBuilder, Value};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use ndarray::Array2;

#[derive(Debug, Deserialize)]
struct PredictRequest { features: String }

#[derive(Debug, Serialize)]
struct PredictResponse { label: i64, confidence: f32 }

struct AppState { session: Arc<Session> }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "model.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[-] Error: {} not found.", model_path);
        std::process::exit(1);
    }

    let session = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(model_path)?;

    let shared_state = Arc::new(AppState { session: Arc::new(session) });

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/predict", post(predict_handler))
        .with_state(shared_state);

    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("[+] Running on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn predict_handler(State(state): State<Arc<AppState>>, Json(payload): Json<PredictRequest>) -> Json<PredictResponse> {
    let input_array = Array2::from_shape_vec((1, 1), vec![payload.features]).unwrap();
    let input_tensor = Value::from_array(state.session.allocator(), &input_array).unwrap();
    let outputs = state.session.run(vec![input_tensor]).unwrap();
    let label = outputs[0].try_extract_tensor::<i64>().unwrap().view()[0];
    Json(PredictResponse { label, confidence: 1.0 })
}
