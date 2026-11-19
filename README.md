# LiDAR Point Cloud Classification — AI Model 3.0 (TerraScan Class Schema)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-purple)
![License](https://img.shields.io/badge/License-BSD--3--Clause-green)
![Version](https://img.shields.io/badge/AI%20Model-3.0.1-orange)

Semantic segmentation of airborne LiDAR point clouds into TerraScan-style classes, built on the open-source [Myria3D](https://github.com/IGNF/myria3d) framework (PDAL preprocessing → Myria3D RandLA-Net inference → rule engine).

AI model 3.0 classifies LiDAR point clouds (`.las` / `.laz`) into the classes of a `.ptc`
class-definition file: Ground, low / medium / high vegetation, buildings and, when the `.ptc`
has them, power-line wires and poles.

This repository is a separate application. It imports nothing from Naksha LiDAR (`gui/`); Naksha
uses it as a library today and will launch it as a separate program (phase 3). Like
Plotline-AI, it will get its own installer with a bundled runtime (phase 4).

## Layout

| Path | Content |
|---|---|
| `src/naksha_ai3/engine/` | The engine: PDAL preprocessing, Myria3D inference, rule engine. Runs as a subprocess in the AI3 runtime. |
| `src/naksha_ai3/job.py` | One job (full file or fences) and batches of jobs. |
| `src/naksha_ai3/reference.py` | Learning from already classified areas inside fences. |
| `src/naksha_ai3/cli.py` | The `naksha-ai3` command line. |
| `src/naksha_ai3/runtime.py`, `controller.py` | Finding the runtime; starting and cancelling the engine. |
| `src/naksha_ai3/ptc.py`, `mapping.py` | Class codes from the `.ptc` file. |
| `src/naksha_ai3/lasio.py`, `fences.py` | LAS/LAZ reading and writing; fence polygons. |
| `tests/` | Tests. `test_standalone_core.py` runs in any Python with numpy/scipy/laspy; `test_engine_step1_attributes.py` needs the AI3 runtime. |

## Runtime

The engine needs its own Python environment (PDAL, torch 2.4, torch-geometric, numpy 1.26).
It is found in this order: `NAKSHA_AI3_PYTHON`; `<app>/runtime/python.exe` (installed app);
`NAKSHA_AI3_RUNTIME` or `%LOCALAPPDATA%\NakshaAI\ai3_runtime` (the environment earlier Naksha
versions set up). Check it with:

```
python -m naksha_ai3 --check
```

## Command line

Run from `src` (or with `src` on `PYTHONPATH`) with the AI3 runtime's Python:

```
python -m naksha_ai3 --input tile.laz --ptc classes.ptc
python -m naksha_ai3 --input a.laz b.laz --ptc classes.ptc --output-dir D:\out
python -m naksha_ai3 --input tile.laz --ptc classes.ptc --fences fences.json --learn
python -m naksha_ai3 --job job.json
```

- **Output:** `<output dir or AI3_Classified next to the input>/<name>_ai3_classified.<ext>` is the
  input file with only the classification replaced, plus `<name>_ai3_report.json`.
- **Fences:** `fences.json` is `{"polygons": [[[x, y], ...], ...]}` in the cloud's XY. Only points
  inside the fences change.
- **`--learn`:** already classified areas inside the fences are kept and used as reference for
  the unfinished points (classes and height bands, and the Ground/Uncategorized percentage).
  Without `--job`, the "already classified" state is the one saved in the file.
- **`--job`:** a launcher's JSON job file. It can also pass the current classification
  (`classes`, `.npy`) and fence point indices (`targets`, `.npy`), so unsaved edits in Naksha
  are used. See `cli.job_from_file`.
- **Machine-readable output:** `PROGRESS <0-100> <message>`, `FILE <i> <n> <path>`,
  `RESULT {json}` per file. Exit codes: 0 done, 1 bad arguments, 2 a file failed, 3 runtime
  unusable, 130 cancelled.

## Large files (tiled full-file runs)

The engine loads its whole input into memory, so a whole file is classified in tiles
(`tiling.py`): one streaming pass splits the file into square tiles of about 8 million points
with a 50 m overlap (in a local work folder, never on the network share), the engine runs on
several tiles at once, and one more streaming pass writes the classified copy. Memory depends on
the tile size, not on the file size, so files of any size can be classified.

A tile gets the same answers a single pass over the whole file would give:

- the model's 50 m receptive fields are anchored on the whole file's grid
  (`NAKSHA_AI3_GRID_ORIGIN`), so every field around a tile's own points holds the same points;
- the building intensity threshold and the point density used for eaves are measured once on
  the whole file during the split;
- whether to apply the building intensity gate is pooled from the first (largest) tiles and then
  fixed for every tile; a first tile that decided otherwise re-runs step 3 only;
- a tile at a sparse corner is re-cut with a wider margin until the engine has enough context.

Ground thinning to Uncategorized stays random per point, as in a single pass.

| Setting | Default | Meaning |
|---|---|---|
| `NAKSHA_AI3_TILE_POINTS` | 8000000 | points per tile |
| `NAKSHA_AI3_TILE_WORKERS` | 2 | tiles classified at the same time (CPU cores, GPU memory, RAM) |
| `NAKSHA_AI3_WORK_DIR` | `%LOCALAPPDATA%\Naksha-AI3\work` | local folder for tiles (needs about the file's size free) |
| `NAKSHA_AI3_INPLACE_MAX_POINTS` (plugin) | 100000000 | above this the plugin opens the classified copy instead of writing into the open view |

## Naksha plugin

`naksha_plugin/` is the AI Model 3.0 plugin for Naksha's Plugins Manager (Install from ZIP).
It bundles `naksha_ai3`, its engine and model, and a required offline conda-pack Python runtime, adds an **AI Model 3.0** button to
the Plugins ribbon, reads the open native (Vulkan/LNT) point cloud with its unsaved edits,
classifies the whole cloud or only selected fences, and writes the result back into the view
with undo. Build the ZIP with:

```
python build_plugin_zip.py --runtime-bundle <bundle-folder>
```

The plugin's own README (`naksha_plugin/README.txt`, shipped in the ZIP) covers install and use.

## Tests

```
python -m pytest tests/test_standalone_core.py
cd tests && <AI3 runtime python> -m unittest test_engine_step1_attributes
```

The plugin automatically checks existing runtimes when loaded after installation and reuses a compatible environment after an engine self-test. If none passes, it installs its own bundled runtime under `%LOCALAPPDATA%/Naksha-AI3/plugin-runtimes/<hash>`. Configured and known AI3 environments are checked as candidates, but are never required. All engine code and model files come from the plugin. The bundle folder supplied to the builder must contain `environment.zip` and a schema-1 `bundle.json` with SHA-256 and unpacked byte count.

## Acknowledgements

This project builds on [Myria3D](https://github.com/IGNF/myria3d) by IGN (Institut National de l'Information Géographique et Forestière), licensed under BSD-3-Clause. The bundled checkpoint `fractal_lidarhd_myria3d.ckpt` is IGN's pretrained LiDAR HD model. If you use Myria3D in research, please cite it as described in its repository.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

## Author

**Likhith R** — GitHub: [mr-liki](https://github.com/mr-liki)
