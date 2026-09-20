# Extensión de navegador: detector de odio en comentarios de YouTube

Marca en la propia página de YouTube los comentarios que el modelo considera posible discurso de odio. Es un
cliente de la API del proyecto (`/predict/batch`): **el modelo no corre en el navegador**.

**Es una estimación y puede equivocarse** (ver las limitaciones en el README principal). El sistema recomienda; no
borra ni denuncia nada.

## Instalación (modo desarrollador)

**No hace falta ninguna cuenta**: ni de Google, ni de YouTube, ni de desarrollador. Tampoco clave de la YouTube
Data API: la extensión lee los comentarios que ya estás viendo en la página. Lo único imprescindible es tener la
API del proyecto en marcha y accesible desde el navegador.

1. Arranca la API (`uvicorn hatedet.api.main:app --port 8000`, o `docker compose up`).
2. En Chrome/Edge abre `chrome://extensions`, activa **Modo desarrollador** y pulsa **Cargar descomprimida**
   seleccionando esta carpeta (`extension/`).
3. Pulsa el icono de la extensión, escribe la URL de la API (por defecto `http://localhost:8000`) y **Probar
   conexión**.
4. Abre cualquier vídeo de YouTube y baja hasta los comentarios.

## Qué hace

- Observa la página (`MutationObserver`) y, según se cargan comentarios, los envía por lotes (máx. 40) a la API.
- Añade una etiqueta a los comentarios en banda **Revisar** o **Posible odio** (opcionalmente también a los «OK»).
- Modos: solo etiqueta · difuminar (clic en la etiqueta para ver) · ocultar «posible odio» y difuminar «revisar».
- Guarda en memoria del navegador el resultado por texto para no repetir peticiones.

## Seguridad y privacidad

- **El texto de los comentarios visibles se envía a la API que configures.** Usa tu propio servidor y HTTPS.
- La URL de la API debe ser **HTTPS** (HTTP solo en `localhost`/`127.0.0.1`); sin usuario ni contraseña en la URL.
- Permisos mínimos: `storage` y acceso a `localhost`. Para un servidor remoto el navegador te pide permiso **solo
  para ese origen** al guardar (`optional_host_permissions`).
- El texto de los comentarios nunca se escribe como HTML (`textContent`, sin `innerHTML`): un comentario con
  etiquetas HTML se muestra literal. El service worker solo acepta mensajes de la propia extensión.
- La clave de API (si la hay) se guarda en `chrome.storage.sync`; no la publiques en repositorios.

## Limitaciones

- Depende de la estructura HTML de YouTube (`ytd-comment-thread-renderer #content-text`), que puede cambiar: si deja
  de marcar comentarios, hay que actualizar el selector en `content.js`.
- No está publicada en ninguna tienda: se carga en modo desarrollador. Para publicarla habría que cumplir las
  políticas de la Chrome Web Store (declarar el uso de datos, política de privacidad).

## Pruebas

- `tests/test_extension_manifest.py`: estructura del manifiesto, permisos mínimos, ausencia de `innerHTML`.
- `extension/test/mock_youtube.html`: página que simula los comentarios de YouTube para probar la extensión contra
  una API local sin necesitar YouTube (servir la carpeta `extension/` con `python -m http.server` y abrir
  `/test/mock_youtube.html` con la API en `localhost:8000`).
