# Create the ConfigMaps
```
kubectl create configmap model-loader-script \
    --from-file=loader.py=scripts/loader.py \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap metrics-exporter-script \
    --from-file=exporter.py=scripts/exporter.py \
    --dry-run=client -o yaml | kubectl apply -f -
```

# Set HF_TOKEN env varibale and Deploy Pod (3 containers: model-loader + vllm + metrics-exporter)...
```
kubectl apply -f manifests/deployment.yaml
kubectl apply -f manifests/service.yaml
```

# Prometheus alerts are optional — only apply if the CRD is installed
```
kubectl apply -f manifests/alerts.yaml
```

# Wait for the deployment
```
kubectl rollout status deployment/qwen-sidecar --timeout=300s
```

# Testing Your Deployment
```
kubectl port-forward svc/qwen-sidecar 8000:8000"
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '\''{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'\''
```

