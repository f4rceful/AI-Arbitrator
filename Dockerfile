FROM rust:1.75-slim as builder
RUN apt-get update && apt-get install -y pkg-config libssl-dev cmake git wget & rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && mv onnxruntime-linux-x64-1.16.3 /opt/onnxruntime
ENV ORT_STRATEGY=system
ENV ORT_LIB_LOCATION=/opt/onnxruntime/lib
WORKDIR /app
COPY service/ .
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libssl3 ca-certificates & rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/onnxruntime/lib/libonnxruntime.so* /usr/lib/
COPY --from=builder /app/target/release/service /app/service
COPY service/model.onnx /app/model.onnx
EXPOSE 3000
CMD ["./service"]
