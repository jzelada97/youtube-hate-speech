# Despliegue en una VM (Linux)

Guía para publicar la API y la demo detrás de **HTTPS**. Necesitas: una VM con Ubuntu/Debian, un dominio (o
subdominio) apuntando a su IP pública y los puertos 80 y 443 abiertos. **No abras el 8000 ni el 8501** al exterior.

## 1. Preparar la VM

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git caddy
sudo usermod -aG docker $USER      # cierra sesión y vuelve a entrar
git clone https://github.com/jzelada97/youtube-hate-speech.git && cd youtube-hate-speech
```

## 2. El modelo (no está en git)

Los modelos entrenados no se versionan (contienen conocimiento derivado de comentarios reales). Dos opciones:

- **Copiarlo desde tu equipo** (recomendado, es lo que evaluaste):
  `scp models/ensemble_v1.joblib models/ensemble_v1.metadata.json usuario@tu-vm:~/youtube-hate-speech/models/`
- **Entrenarlo en la VM**: descarga `youtoxic_english_1000.csv` en `data/raw/` y ejecuta, con un entorno virtual,
  `pip install -e ".[serve]" && python scripts/train.py --model ensemble`.

## 3. Configuración

```bash
cp .env.example .env
# Edita .env: define una HATEDET_API_KEY larga y aleatoria (openssl rand -hex 32) y, si quieres usar la API oficial
# de YouTube, YOUTUBE_API_KEY. Añade el origen de tu extensión/página a HATEDET_CORS_REGEX si hace falta.
```

## 4. Arrancar los contenedores

```bash
docker compose --env-file .env up -d --build
docker compose ps            # api debe aparecer "healthy"
curl -s http://127.0.0.1:8000/health
```

La API y la demo escuchan solo en `127.0.0.1` (ver `docker-compose.yml`).

## 5. HTTPS con Caddy

Edita `deploy/Caddyfile` (cambia `hate.tudominio.com` por tu dominio) y:

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
curl -s https://hate.tudominio.com/health
```

Caddy obtiene y renueva solo el certificado (Let's Encrypt). La API queda en `https://hate.tudominio.com` y la demo
en `https://hate.tudominio.com/demo/`.

## 6. Arranque automático y actualizaciones

Los contenedores tienen `restart: unless-stopped`, así que vuelven solos tras un reinicio de la VM. Para actualizar:

```bash
git pull && docker compose --env-file .env up -d --build
```

## 7. Comprobaciones de seguridad

- `HATEDET_API_KEY` definida (sin ella cualquiera podría usar tu API): `curl -i https://hate.tudominio.com/predict
  -X POST -H 'Content-Type: application/json' -d '{"text":"hola"}'` debe devolver **401**.
- Firewall: solo 22, 80 y 443 (`sudo ufw allow 22,80,443/tcp && sudo ufw enable`).
- Los comentarios se envían a tu servidor: la API **no registra su texto**; revisa que los logs de Caddy no lo
  guarden (`log` sin `body`). Informa a los usuarios de la extensión (datos personales, RGPD).
- Extensión: en su popup pon la URL `https://hate.tudominio.com` y la clave de API.

## Si algo falla

| Síntoma | Causa probable |
|---|---|
| `api` no llega a *healthy* | Falta el fichero del modelo en `models/` o `HATEDET_MODEL_PATH` apunta mal (`docker compose logs api`). |
| 401 en todas las peticiones | La clave de la extensión/cliente no coincide con `HATEDET_API_KEY`. |
| Error CORS desde una web | Añade su origen a `HATEDET_CORS_REGEX`. |
| `/analyze/video` devuelve 502 | YouTube cambió su HTML (actualiza `youtube-comment-downloader`) o no se puede acceder al vídeo; con `YOUTUBE_API_KEY` se usa la API oficial. |
| Certificado no se emite | El dominio no apunta a la IP de la VM o los puertos 80/443 están cerrados. |
