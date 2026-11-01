import { contextBridge, ipcRenderer } from "electron";

const api = {
  openFileDialog: (): Promise<string | null> =>
    ipcRenderer.invoke("dialog:openFile"),

  openExternalViewer: (niftiPath: string): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke("viewer:openExternal", niftiPath),

  // API base URL (FastAPI backend)
  apiBase: `http://127.0.0.1:${18923}`,
};

contextBridge.exposeInMainWorld("diffsuite", api);

export type DiffSuiteAPI = typeof api;
