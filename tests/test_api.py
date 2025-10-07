import requests

import pytest

def test_health_endpoint():
	"""Testea que el endpoint /health responde correctamente."""
	response = requests.get("http://localhost:8000/health")
	assert response.status_code == 200
	assert response.json().get("status") == "healthy"


def test_pytest_raises():
	"""Ejemplo de uso de pytest.raises para validar excepciones."""
	with pytest.raises(ZeroDivisionError):
		_ = 1 / 0
# Pruebas para la API