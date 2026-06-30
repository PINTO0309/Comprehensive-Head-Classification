import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronBenchmark', {
  finishCliBenchmark(payload: unknown) {
    ipcRenderer.send('benchmark-finished', payload);
  },
});
