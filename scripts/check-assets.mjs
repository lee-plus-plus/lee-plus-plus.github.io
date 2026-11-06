// Run with Node.js; no npm dependencies or build step are required.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = process.argv[2] ? path.resolve(process.argv[2]) : path.join(project, 'preview');
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(root, 'site-data.js'), 'utf8'), context);
const { profile, papers } = context.window.ACADEMIC_DATA;
const references = new Set();

function addReference(value) {
  if (/^(?:[a-z]+:|#|\/\/)/i.test(value)) return;
  references.add(path.resolve(root, value.split(/[?#]/)[0]));
}

addReference('assets/' + profile.avatar);
addReference('assets/' + profile.avatarAlt);
for (const paper of papers) addReference('assets/' + paper.image);
addReference('index.html');
for (const name of ['index.html']) {
  const html = fs.readFileSync(path.join(root, name), 'utf8');
  for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) addReference(match[1]);
}
for (const name of ['style.css']) {
  const css = fs.readFileSync(path.join(root, name), 'utf8');
  for (const match of css.matchAll(/url\(["']?([^"')]+)["']?\)/g)) addReference(match[1]);
}

function filesIn(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const filename = path.join(directory, entry.name);
    return entry.isDirectory() ? filesIn(filename) : [filename];
  });
}

const siteFiles = filesIn(root);
const assets = siteFiles.filter(filename => filename.startsWith(path.join(root, 'assets') + path.sep));
const relative = filename => path.relative(root, filename).replaceAll('\\', '/');
const unused = siteFiles.filter(filename => !references.has(filename) && path.basename(filename) !== 'PTSerif-OFL.txt');
const missing = [...references].filter(filename => !fs.existsSync(filename));
const byHash = new Map();
for (const filename of assets) {
  const hash = crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
  byHash.set(hash, [...(byHash.get(hash) || []), relative(filename)]);
}

console.log(JSON.stringify({
  siteFileCount: siteFiles.length,
  assetCount: assets.length,
  assetBytes: assets.reduce((total, filename) => total + fs.statSync(filename).size, 0),
  missing: missing.map(relative),
  unused: unused.map(relative),
  duplicates: [...byHash.values()].filter(filenames => filenames.length > 1),
}, null, 2));
if (missing.length || unused.length) process.exitCode = 1;
