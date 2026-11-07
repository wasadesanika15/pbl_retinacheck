import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = join(projectDir, "dist");
const vendorDir = join(projectDir, "static", "vendor");
const tflitePackageDir = join(projectDir, "node_modules", "@tensorflow", "tfjs-tflite");

const runtimeFiles = [
  ["@tensorflow/tfjs-core/dist/tf-core.min.js", "tf-core.min.js"],
  [
    "@tensorflow/tfjs-backend-cpu/dist/tf-backend-cpu.min.js",
    "tf-backend-cpu.min.js",
  ],
  ["@tensorflow/tfjs-tflite/dist/tf-tflite.min.js", "tf-tflite.min.js"],
];

await rm(outputDir, { recursive: true, force: true });
await rm(vendorDir, { recursive: true, force: true });
await mkdir(vendorDir, { recursive: true });

for (const [source, destination] of runtimeFiles) {
  await cp(
    join(projectDir, "node_modules", source),
    join(vendorDir, destination),
  );
}

await cp(join(tflitePackageDir, "wasm"), vendorDir, {
  recursive: true,
});
await cp(join(projectDir, "static"), join(outputDir, "static"), {
  recursive: true,
});
await writeFile(
  join(outputDir, "index.html"),
  await readFile(join(projectDir, "templates", "index.html")),
);

console.log(`Built static site in ${outputDir}`);
