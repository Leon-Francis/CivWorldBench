# Engine Builds

Build from this directory with Docker and network access:

```sh
docker build --build-arg TASK_PERSIST=0 -t wm/freeciv-server:3.1.6-v2 .
docker build --build-arg TASK_PERSIST=1 -t wm/freeciv-server:3.1.6-v3 .
```

The benchmark corpus used v2. The per-turn agent-play study used v3, which also
persists AI unit tasks across saved-state reloads. Both use Freeciv 3.1.6 and the
same v2 behavior and ruleset patches. The Dockerfile verifies the downloaded
upstream archive's SHA256 before extracting it.

Historical local image configuration digests:

- v2: `sha256:3b373800b97e6eaf8b16f9cf965b3519a75ea9f6a19c99dcf354053e66440e70`
- v3: `sha256:19c25fd2cf4d6155318a603da4876115c1e2672f01293da4a6375d4b627919f7`

These are provenance identifiers, not pullable registry digests. Rebuilding
against current Debian packages may produce different image bytes. No image
binary is included, and fresh container builds were not run for this compact
release. The frozen questions and numerical analyses do not require Docker.
