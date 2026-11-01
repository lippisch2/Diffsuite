import { app, BrowserWindow, ipcMain, dialog } from "electron";
import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as fs from "fs";
import fetch from "node-fetch";

const API_PORT = 18923;
const API_URL = `http://127.0.0.1:${API_PORT}`;

let mainWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;

/**
 * Start the FastAPI backend as a subprocess.
 * Uses the active Python interpreter (same env as `diffsuite` CLI).
 */
function startPythonBackend(): Promise<void> {
  return new Promise((resolve, reject) => {
    const pythonCmd =
      process.platform === "win32" ? "python" : "python3";

    pythonProcess = spawn(
      pythonCmd,
      [
        "-m",
        "uvicorn",
        "diffsuite.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        String(API_PORT),
      ],
      {
        stdio: ["ignore", "pipe", "pipe"],
        env: { ...process.env },
      }
    );

    pythonProcess.stdout?.on("data", (d) => {
      console.log(`[uvicorn] ${d.toString().trim()}`);
    });
    pythonProcess.stderr?.on("data", (d) => {
      console.error(`[uvicorn:err] ${d.toString().trim()}`);
    });

    pythonProcess.on("error", (err) => {
      console.error("[uvicorn] Failed to start:", err.message);
      reject(err);
    });

    // Poll /api/status until ready
    const start = Date.now();
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/api/status`);
        if (res.ok) {
          clearInterval(poll);
          console.log("[diffsuite] FastAPI backend ready");
          resolve();
        }
      } catch {
        if (Date.now() - start > 30000) {
          clearInterval(poll);
          reject(new Error("FastAPI backend failed to start within 30s"));
        }
      }
    }, 500);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: "#1e1e1e",
    webPreferences: {
      preload: path.join(__dirname, "../preload/preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    mainWindow.loadFile(
      path.join(__dirname, "../renderer/index.html")
    );
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ── IPC handlers ─────────────────────────────────────────────────────────────
ipcMain.handle("dialog:openFile", async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile"],
    filters: [
      { name: "NIfTI / DICOM", extensions: ["nii", "nii.gz", "gz", "dcm"] },
      { name: "NIfTI", extensions: ["nii", "nii.gz", "gz"] },
      { name: "DICOM", extensions: ["dcm"] },
      { name: "All", extensions: ["*"] },
    ],
  });
  if (result.canceled) return null;
  return result.filePaths[0];
});

ipcMain.handle(
  "viewer:openExternal",
  async (_event, niftiPath: string) => {
    const niviewer = path.join(
      app.getAppPath(),
      "..",
      "..",
      "niviewer.py"
    );
    // Fallback: try project root relative to cwd
    const fallback = path.join(process.cwd(), "niviewer.py");
    const target = fs.existsSync(niviewer) ? niviewer : fallback;
    const pythonCmd =
      process.platform === "win32" ? "python" : "python3";
    try {
      spawn(pythonCmd, [target, niftiPath], { detached: true, stdio: "ignore" });
      return { ok: true };
    } catch (e: any) {
      return { ok: false, error: e.message };
    }
  }
);

app.whenReady().then(async () => {
  try {
    await startPythonBackend();
  } catch (e) {
    console.error("Backend start error:", e);
    app.quit();
    return;
  }
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
});
