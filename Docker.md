# Instrucciones esenciales para Docker

1. Construir la imagen:
```pwsh docker build -t mba-api:prod .
```

2. Ejecutar el container:
```pwsh docker run --rm -p 8000:8000 mba-api:prod
```

3. Probar endpoint de salud:
```pwsh curl http://localhost:8000/health
```

4. Detener el container (Ctrl+C en la terminal donde corre):
No requiere comando extra si usas --rm.

