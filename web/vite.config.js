import { defineConfig } from "vite";

export default defineConfig({
    base: "gbcolorize",
    optimizeDeps: {
        exclude: ["onnxruntime-web"],
    },
    assetsInclude: ["**/*.onnx"],
    root: "src",
    build: {
        outDir: "../dist",
        rollupOptions: {
            input: "src/index.html",
        },
    },
});
