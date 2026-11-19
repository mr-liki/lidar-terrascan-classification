AI Model 3.0 - Naksha plugin
============================

Install
  Plugins > Manager > Install from ZIP > choose AI_Model_3_Plugin_v3.0.1.zip > Install Plugin.
  An "AI Model 3.0" button appears in the Plugins ribbon under Custom Plugins.

Requirements
  - A LAS/LAZ open in the native Vulkan view (a single file, not a collection).
  - The class-definition .ptc (applied in Display Mode, or chosen in the plugin window).
  - The ZIP includes its own Python runtime and all AI dependencies. No separate Python,
    AI3 installation, internet connection or host AI setup is required.
  - Installation automatically checks existing Python environments. A compatible, verified
    AI3 runtime is reused. Otherwise the bundled runtime is installed offline in a private per-user folder:
    %LOCALAPPDATA%\Naksha-AI3\plugin-runtimes\<bundle hash>\env
    Allow several minutes and at least 7 GiB free there, in addition to plugin storage.
    Setup starts in the background when the plugin loads after installation. Run waits for
    setup to finish; interrupted setup is retried automatically.
  - The host must support the native Vulkan/LNT view and LNT AI bridge. Its plugin manager
    supplies the UI; classification uses the plugin's own environment.
  - An NVIDIA GPU is recommended; without one the engine runs on the CPU, slowly.

Use
  1. Click AI Model 3.0.
  2. Check the .ptc and the class codes shown under it.
  3. Choose "The whole point cloud" or "Only inside the selected fences" and tick the fences
     (they are highlighted in yellow in the view).
  4. Optional: "Learn from already classified areas inside the fences" keeps your classified
     areas unchanged and makes the unfinished points follow their classes and height bands.
  5. Run. The result is written into the view (Ctrl+Z undoes it) and a classified copy of the
     source is saved in AI3_Classified next to it, with a JSON report.

Large point clouds
  A whole cloud is classified in tiles of about 8 million points, several at a time ("tiles
  classified at the same time" in Options; 2 suits a 6 GB GPU). Memory depends on the tile size,
  so clouds of any size work; tiles are written to %LOCALAPPDATA%\Naksha-AI3\work, which needs
  about the file's size free.
  - Up to 100 million points the result is written into the open view (Ctrl+Z undoes it).
  - Larger clouds are not read into memory: when classification finishes, the classified copy
    is opened in the view (your drawings stay). The original file is not changed.

Contents
  plugin.py, naksha_ai3_host.py, naksha_ai3_window.py   the plugin
  naksha_ai3/                                           the AI3 package, engine and model

runtime/environment.zip, runtime/bundle.json             offline Python environment
