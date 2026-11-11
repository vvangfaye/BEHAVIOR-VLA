# Local organizer rehearsal: checkpoint 1688977

This directory is separate from the submitted 1,000-episode results. No submission
field or original result is modified by this preparation or rehearsal.

## Status

Read `SETUP_STATUS.json`, `RUN_STATE.json` (once created), and `runtime/logs/continuation.log`.
The continuation waits at most four hours for preparation, then at most six hours
for a free GPU. A failed preparation stage requires inspection; it does not start
a simulator with incomplete inputs.

The prepared run is `turning_on_radio`, public instance 301, rollout 0, whole-task
conditioning, the default wrapper, no oracle, no shortened step limit. It writes
JSON, logs and complete video into `episode_turning_on_radio_301/`.

One GPU must have no compute processes, zero reported utilization, and at least
44,000 MiB free for three observations. The admission condition is checked again
immediately before startup. The continuation never kills or pauses someone
else's task. It will not start a second episode once `ONE_EPISODE_STARTED.json`
exists.

## Provenance and isolation

- The 1,461 evaluator source files were extracted from the original submitted ZIP
  and verified against `REPRODUCTION_SOURCE_SHA256.json`.
- Official asset versions and hashes are in `OFFICIAL_DATASETS.json`. The existing
  private AOSS copies have the same hashes. Credentials are not copied from the Mac.
- The original evaluator archive's size and ETag match the historical object.
  Its bytes are checked against OSS's full-object CRC64, and a new full SHA256
  is recorded before extraction. Multipart upload boundaries are not assumed.
- The submitted Docker archive is checked against its original SHA256 and its
  configuration against the public image. Every uncompressed layer is checked
  against the configuration's diff IDs before execution.
- Because the data disk is NTFS, the image is stored and executed with a rootless
  OCI runtime rather than moving or reconfiguring the shared Docker daemon.
  Single-user ownership mapping differs from a rootful Docker extraction; the
  image file contents and application entrypoint are preserved.
- Policy and evaluator communicate inside a private loopback network namespace.
  No listening port is published to the host or LAN. Only the selected GPU and its matching
  rendering device are exposed to that namespace.

`*.private.json` contain expiring download capabilities. Do not publish them.
This is a 48GB RTX 6000 Ada rehearsal, not physical 24GB compatibility testing.

## Preparation repairs

The corrupt robot archive was quarantined and downloaded again; its official
SHA256 now matches. The environment archive matches OSS's full-object CRC64.
Warp 1.12.0 was added, setuptools was restored to 80.9.0, and packaging 25.0 was
reinstalled to repair inconsistent files in the source environment.

Execution uses `/mnt/data1/faye/b1k-rehearsal-1688977`, a symlink to this directory,
so Torch's compiler does not split paths at the project name's space. Python's
build metadata and environment library search path use that alias. Original
build metadata is retained under `runtime/patch-backups/`. Both the host CLI
preflight and the CLI inside the isolated namespace passed without GPU devices.
